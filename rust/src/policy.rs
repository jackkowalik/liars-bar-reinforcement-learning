use std::collections::HashMap;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::{AcqRel, Acquire, Relaxed}};

use crate::engine::{Game, MAX_ACTIONS};

// Call info set:
//   n_alive, claim size, matching cards I can't see (8 - in my hand - I played, capped at 6),
//   claimer's total claims this round, third-party claims this round, claimer cards left,
//   my matching, my non-matching, my pulls, claimer pulls, other players still holding cards.
//   plus the claimer's lie record this game (lie_bucket), and whether this claim could be
//   the Devil (0 not in play, 1 impossible because I hold or played it, 2 possible).
// The claimer's own claims and third-party claims are kept apart so "the claimer's story
// can't all be true" is distinguishable from "someone else lied earlier".
// ...plus the claimer's own calling rate (call_bucket), which matters most in 1v1, where the
// claimer is also the one who would catch my bluff next.
pub const CALL_DIMS: [u64; 14] = [3, 3, 7, 5, 6, 5, 6, 6, 6, 6, 3, 4, 3, 6];
// Play info set: n_alive, my matching, my non-matching, next player's cards, my pulls,
// next player's pulls, unaccounted matching cards, round claims bucket, other players with cards,
// next player's calling record this game (call_bucket), Devil state (0 not in play, 1 in my
// hand, 2 I played it, 3 elsewhere or undealt), and my own caught-lying record (my table image).
// ...plus how accurate the next player's calls have been (call_accuracy_bucket).
pub const PLAY_DIMS: [u64; 13] = [3, 6, 6, 6, 6, 6, 8, 4, 3, 6, 4, 4, 4];

/// How often `q` was caught lying when called this game:
/// 0 fewer than `min` reveals (3, or 2 with fast reads), 1 mostly honest, 2 mixed, 3 mostly
/// lying (smoothed (l+1)/(c+2)).
#[inline]
pub fn lie_bucket_min(g: &Game, q: usize, min: f32) -> u64 {
    let c = g.rec_called[q] as f32;
    if c < min {
        return 0;
    }
    let r = (g.rec_lies[q] as f32 + 1.0) / (c + 2.0);
    if r < 0.4 { 1 } else if r > 0.6 { 3 } else { 2 }
}

/// How often `q` calls when they can this game: 0 fewer than `min` chances (5, or 3 with fast
/// reads), then five levels from rarely to very often.
#[inline]
pub fn call_bucket_min(g: &Game, q: usize, min: f32) -> u64 {
    let o = g.rec_opps[q] as f32;
    if o < min {
        return 0;
    }
    // Real lobbies average ~24% (openers ~11%); the steps separate passive players from
    // Grand_Master-style heavy callers.
    let r = g.rec_calls[q] as f32 / o;
    match r {
        r if r < 0.10 => 1,
        r if r < 0.20 => 2,
        r if r < 0.30 => 3,
        r if r < 0.45 => 4,
        _ => 5,
    }
}

/// Which info-set features are on. Saved in the policy file (ablation switches).
#[derive(Clone, Copy, Debug)]
pub struct Feat {
    pub history: bool,
    pub self_image: bool,
    /// Call rates in the old 3 levels (rare / sometimes / often) instead of 5.
    pub coarse_calls: bool,
    /// The claimer's calling rate in the call decision.
    pub claimer_calls: bool,
    /// The next player's call accuracy in the play decision.
    pub call_accuracy: bool,
    /// Play decisions also see the player after next: their cards and revolver pulls.
    pub after_next: bool,
    /// Play decisions also see the threat picture: is the next player the leader (fewest pulls
    /// among my opponents), and how close the most endangered opponent is to dying.
    pub threat: bool,
    /// Play decisions see the whole table: for each other player in seat order from me (next,
    /// second, third), cards left (none / 1-3, can empty next turn / 4-5) and pulls (0-1 / 2-3 /
    /// 4-5). Replaces the next player's detailed card and pull fields with these buckets.
    pub full_table: bool,
    /// Read players sooner: lying record after 2 reveals (not 3), calling habit after 3
    /// chances (not 5). Noisier, but catches heavy callers before they cost several pulls.
    pub fast_reads: bool,
    /// Call decisions see the table too: the card-count odds the claim is true (all of the
    /// claimer's claims this round / just this play), in place of the two raw count fields, and
    /// each other player's cards left and pulls in seat order from me.
    pub call_table: bool,
}

impl Feat {
    #[inline]
    fn lie_b(&self, g: &Game, q: usize) -> u64 {
        lie_bucket_min(g, q, if self.fast_reads { 2.0 } else { 3.0 })
    }

    pub fn full(history: bool, self_image: bool) -> Feat {
        Feat { history, self_image, coarse_calls: false, claimer_calls: true, call_accuracy: true, after_next: false, threat: false, full_table: false, fast_reads: false, call_table: false }
    }
}

/// Call-rate bucket honouring the coarse switch (old thresholds mapped onto the same codes).
#[inline]
pub fn call_bucket_f(g: &Game, q: usize, f: &Feat) -> u64 {
    if !f.coarse_calls {
        return call_bucket_min(g, q, if f.fast_reads { 3.0 } else { 5.0 });
    }
    let o = g.rec_opps[q] as f32;
    if o < (if f.fast_reads { 3.0 } else { 5.0 }) {
        return 0;
    }
    let r = g.rec_calls[q] as f32 / o;
    if r < 0.15 { 1 } else if r > 0.4 { 5 } else { 3 }
}

/// How often `q`'s voluntary calls exposed a lie: 0 fewer than 3 calls, 1 under half right,
/// 2 half to three quarters, 3 more (they read the player before them well).
#[inline]
pub fn call_accuracy_bucket(g: &Game, q: usize) -> u64 {
    let c = g.rec_calls[q] as u32;
    if c < 3 {
        return 0;
    }
    let right = g.rec_calls_right[q] as u32;
    if right * 2 < c { 1 } else if right * 4 <= c * 3 { 2 } else { 3 }
}
pub const CALL_STRIDE: usize = 2;
pub const PLAY_STRIDE: usize = MAX_ACTIONS;

#[inline]
fn mix(dims: &[u64], vals: &[u64]) -> u64 {
    let mut k = 0;
    for (d, v) in dims.iter().zip(vals) {
        debug_assert!(v < d, "key field out of range: {v} >= {d}");
        k = k * d + v;
    }
    k
}

pub fn call_index(f: [u64; 14]) -> u64 {
    mix(&CALL_DIMS, &f)
}

pub fn play_index(f: [u64; 13]) -> u64 {
    mix(&PLAY_DIMS, &f)
}

/// Matching cards that could still be in other hands minus what others claimed this round.
/// Negative means someone has definitely lied this round.
#[inline]
pub fn slack(g: &Game, p: usize) -> i32 {
    let others_claimed = (g.claims_total - g.claims[p]) as i32;
    g.truths() as i32 - g.hm[p] as i32 - g.played_match[p] as i32 - others_claimed
}

pub fn call_fields(g: &Game, p: usize, f: &Feat) -> [u64; 14] {
    let history = f.history;
    let lp = g.last.expect("call key without a play");
    let prev = lp.player;
    let unseen = g.truths().saturating_sub(g.hm[p] + g.played_match[p]);
    let devil = if !g.devil_active { 0 } else if g.hd[p] > 0 || g.played_devil[p] { 1 } else { 2 };
    let third = g.claims_total - g.claims[p] - g.claims[prev];
    [
        g.n_alive() as u64 - 2,
        lp.n as u64 - 1,
        unseen.min(6) as u64,
        g.claims[prev].min(5) as u64 - 1,
        third.min(5) as u64,
        g.hand(prev).min(4) as u64,
        g.hm[p] as u64,
        g.hn[p] as u64,
        g.shots[p] as u64,
        g.shots[prev] as u64,
        g.others_active(p, prev).min(2) as u64,
        if history { f.lie_b(g, prev) } else { 0 },
        devil,
        if history && f.claimer_calls { call_bucket_f(g, prev, f) } else { 0 },
    ]
}

// Devil's Deal info set (stored in the call table, offset past every call key): n_alive,
// my true cards in the window (0 none possible, 1 maybe, 2 certainly), my cards in the window,
// other players in the window, unaccounted true cards, card-count slack, my pulls, highest
// other pulls, Devil state.
pub const DEAL_DIMS: [u64; 9] = [3, 3, 5, 4, 9, 8, 6, 6, 3];
const DEAL_BASE: u64 = 1 << 50;

pub fn deal_fields(g: &Game, p: usize) -> [u64; 9] {
    let (mut mine, mut maybe_true, mut sure_true) = (0u8, false, false);
    let mut others = [false; crate::engine::MAX_PLAYERS];
    for (pl, k) in g.deal_window() {
        if pl.player == p {
            mine += k;
            let trues = pl.n_true + pl.devil as u8;
            maybe_true |= trues > 0;
            sure_true |= k + trues > pl.n;
        } else {
            others[pl.player] = true;
        }
    }
    let unseen = g.truths().saturating_sub(g.hm[p] + g.played_match[p]);
    let devil = if !g.devil_active { 0 } else if g.hd[p] > 0 || g.played_devil[p] { 1 } else { 2 };
    let max_other = (0..g.n).filter(|&q| q != p && g.alive[q]).map(|q| g.shots[q]).max().unwrap_or(0);
    [
        g.n_alive() as u64 - 2,
        if sure_true { 2 } else if maybe_true { 1 } else { 0 },
        mine as u64,
        others.iter().filter(|&&o| o).count().min(3) as u64,
        unseen.min(8) as u64,
        (slack(g, p).clamp(-3, 4) + 3) as u64,
        g.shots[p] as u64,
        max_other as u64,
        devil,
    ]
}

pub fn deal_key(g: &Game, p: usize) -> u64 {
    DEAL_BASE + mix(&DEAL_DIMS, &deal_fields(g, p))
}

pub fn play_fields(g: &Game, p: usize, f: &Feat) -> [u64; 13] {
    let (history, self_image) = (f.history, f.self_image);
    let nxt = g.next_active(p);
    let (next_left, next_shots) = match nxt {
        Some(q) => (g.hand(q) as u64, g.shots[q] as u64),
        None => (0, 0),
    };
    let total_b = match g.claims_total {
        0 => 0,
        1..=2 => 1,
        3..=4 => 2,
        _ => 3,
    };
    [
        g.n_alive() as u64 - 2,
        g.hm[p] as u64,
        g.hn[p] as u64,
        next_left,
        g.shots[p] as u64,
        next_shots,
        (slack(g, p).clamp(-1, 6) + 1) as u64,
        total_b,
        g.others_active(p, nxt.unwrap_or(p)).min(2) as u64,
        if history { nxt.map_or(0, |q| call_bucket_f(g, q, f)) } else { 0 },
        if !g.devil_active { 0 } else if g.hd[p] > 0 { 1 } else if g.played_devil[p] { 2 } else { 3 },
        if self_image { f.lie_b(g, p) } else { 0 },
        if history && f.call_accuracy { nxt.map_or(0, |q| call_accuracy_bucket(g, q)) } else { 0 },
    ]
}

#[inline]
pub fn call_key(g: &Game, p: usize, f: &Feat) -> u64 {
    if !f.call_table {
        return call_index(call_fields(g, p, f));
    }
    let mut fields = call_fields(g, p, f);
    fields[2] = 0; // unaccounted true cards and the third parties' claims: replaced by the odds
    fields[4] = 0;
    let lp = g.last.expect("call key without a play");
    let unseen = g.truths().saturating_sub(g.hm[p] + g.played_match[p]) as u64;
    let pop = (crate::engine::DECK_SIZE - crate::engine::HAND_SIZE) as u64;
    let hs = crate::engine::HAND_SIZE as u64;
    let all = crate::agents::hypergeom_at_least(g.claims[lp.player] as u64, pop, unseen, hs);
    let this = crate::agents::hypergeom_at_least(lp.n as u64, pop, unseen, hs).max(all);
    let all_b = if all < 0.1 { 0 } else if all < 0.4 { 1 } else if all < 0.75 { 2 } else { 3 };
    let this_b = if this < 0.4 { 0 } else if this < 0.8 { 1 } else { 2 };
    let mut k = call_index(fields) * 12 + all_b * 3 + this_b;
    // The other players (not me, not the claimer) in seat order from me: cards 0 / 1-3 / 4-5,
    // pulls 0-1 / 2-3 / 4-5; a missing seat counts as no cards.
    let mut q = p;
    let mut seen = 0;
    for _ in 0..g.n {
        q = (q + 1) % g.n;
        if q == p || seen == 2 {
            break;
        }
        if q == lp.player || !g.alive[q] {
            continue;
        }
        let h = g.hand(q);
        let cards = if h == 0 { 0 } else if h <= 3 { 1 } else { 2 };
        k = k * 9 + cards * 3 + (g.shots[q] / 2).min(2) as u64;
        seen += 1;
    }
    for _ in seen..2 {
        k *= 9;
    }
    k
}

#[inline]
pub fn play_key(g: &Game, p: usize, f: &Feat) -> u64 {
    let mut k = if f.full_table {
        let mut fields = play_fields(g, p, f);
        fields[3] = 0; // next player's cards and pulls: covered by the table summary below
        fields[5] = 0;
        let mut k = play_index(fields);
        let mut q = p;
        for _ in 0..3 {
            let (cards, pulls) = match g.next_active(q).filter(|&r| r != p) {
                Some(r) => {
                    q = r;
                    let h = g.hand(r);
                    (if h == 0 { 0 } else if h <= 3 { 1 } else { 2 }, (g.shots[r] / 2).min(2) as u64)
                }
                None => (0, 0),
            };
            k = k * 9 + cards * 3 + pulls;
        }
        k
    } else {
        play_index(play_fields(g, p, f))
    };
    if f.threat {
        // Next player vs my other opponents: 0 the leader (fewest pulls), 1 tied, 2 behind.
        // Most endangered opponent: pulls 0-2 / 3 / 4-5. Appended like after_next.
        let nxt = g.next_active(p);
        let others = || (0..g.n).filter(move |&q| q != p && g.alive[q]);
        let lead = nxt.map_or(0, |q| {
            let rest = others().filter(|&r| r != q).map(|r| g.shots[r]).min();
            match rest {
                Some(m) if g.shots[q] < m => 0,
                Some(m) if g.shots[q] == m => 1,
                Some(_) => 2,
                None => 0,
            }
        });
        let worst = others().map(|q| g.shots[q]).max().unwrap_or(0);
        k = k * 9 + lead * 3 + if worst <= 2 { 0 } else if worst == 3 { 1 } else { 2 };
    }
    if !f.after_next {
        return k;
    }
    // The player after next (the one my play's receiver passes the danger to): cards 0 / 1-2 / 3+,
    // pulls 0-1 / 2-3 / 4-5. Appended so keys without it keep their old values.
    let after = g.next_active(p).and_then(|q| g.next_active(q)).filter(|&r| r != p);
    let (cards, pulls) = after.map_or((0, 0), |r| {
        let h = g.hand(r);
        (if h == 0 { 0 } else if h <= 2 { 1 } else { 2 }, (g.shots[r] / 2).min(2) as u64)
    });
    k * 9 + cards * 3 + pulls
}

#[inline]
fn hash64(k: u64) -> u64 {
    let mut x = k.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Regret / average-strategy table keyed by info-set id, shared across threads (hogwild).
/// Open addressing with linear probing; slots are claimed with a CAS on the key so concurrent
/// inserts are safe. Values use relaxed atomics, which compile to plain loads and stores on
/// x86; the occasional lost update from a race is noise next to outcome-sampling variance.
pub struct Table {
    pub stride: usize,
    mask: usize,
    keys: Vec<AtomicU64>, // key + 1, 0 = empty
    len: AtomicU64,
    regret: Vec<AtomicU32>,
    strat: Vec<AtomicU64>,
}

impl Table {
    pub fn new(log2_cap: u32, stride: usize) -> Table {
        let cap = 1usize << log2_cap;
        Table {
            stride,
            mask: cap - 1,
            keys: (0..cap).map(|_| AtomicU64::new(0)).collect(),
            len: AtomicU64::new(0),
            regret: (0..cap * stride).map(|_| AtomicU32::new(0)).collect(),
            strat: (0..cap * stride).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.len.load(Relaxed) as usize
    }

    pub fn capacity(&self) -> usize {
        self.mask + 1
    }

    /// Slot for `key`, inserting it if absent.
    #[inline]
    pub fn slot(&self, key: u64) -> usize {
        let tag = key + 1;
        let mut i = hash64(key) as usize & self.mask;
        loop {
            let k = self.keys[i].load(Acquire);
            if k == tag {
                return i;
            }
            if k == 0 {
                match self.keys[i].compare_exchange(0, tag, AcqRel, Acquire) {
                    Ok(_) => {
                        let n = self.len.fetch_add(1, Relaxed) + 1;
                        if n as usize * 10 > self.capacity() * 9 {
                            panic!("info-set table full ({n} entries); raise --table-log");
                        }
                        return i;
                    }
                    Err(k2) if k2 == tag => return i,
                    Err(_) => {}
                }
            }
            i = (i + 1) & self.mask;
        }
    }

    #[inline]
    pub fn r(&self, slot: usize, a: usize) -> f32 {
        f32::from_bits(self.regret[slot * self.stride + a].load(Relaxed))
    }

    #[inline]
    pub fn set_r(&self, slot: usize, a: usize, v: f32) {
        self.regret[slot * self.stride + a].store(v.to_bits(), Relaxed);
    }

    #[inline]
    fn s(&self, slot: usize, a: usize) -> f64 {
        f64::from_bits(self.strat[slot * self.stride + a].load(Relaxed))
    }

    /// Current regret-matching strategy.
    #[inline]
    pub fn sigma(&self, slot: usize, n: usize, out: &mut [f32; MAX_ACTIONS]) {
        let mut sum = 0.0f32;
        for k in 0..n {
            let v = self.r(slot, k);
            out[k] = v;
            sum += v;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for v in out.iter_mut().take(n) {
                *v *= inv;
            }
        } else {
            let u = 1.0 / n as f32;
            for v in out.iter_mut().take(n) {
                *v = u;
            }
        }
    }

    #[inline]
    pub fn add_strat(&self, slot: usize, n: usize, sig: &[f32; MAX_ACTIONS], w: f64) {
        for k in 0..n {
            let a = &self.strat[slot * self.stride + k];
            let v = f64::from_bits(a.load(Relaxed)) + w * sig[k] as f64;
            a.store(v.to_bits(), Relaxed);
        }
    }

    /// Slot holding `key`, if present (no insert).
    pub(crate) fn find(&self, key: u64) -> Option<usize> {
        let tag = key + 1;
        let mut i = hash64(key) as usize & self.mask;
        loop {
            match self.keys[i].load(Acquire) {
                0 => return None,
                k if k == tag => return Some(i),
                _ => i = (i + 1) & self.mask,
            }
        }
    }

    /// Adds `worker - base` into this table, for every info set the worker has. Merging
    /// independent runs that all resumed from `base` this way combines their learning.
    pub(crate) fn add_delta(&self, worker: &Table, base: &Table) {
        for (ws, key) in worker.occupied() {
            let ms = self.slot(key);
            let bs = base.find(key);
            for j in 0..self.stride {
                let (br, bst) = bs.map_or((0.0, 0.0), |b| (base.r(b, j), base.s(b, j)));
                self.set_r(ms, j, self.r(ms, j) + worker.r(ws, j) - br);
                let a = &self.strat[ms * self.stride + j];
                a.store((self.s(ms, j) + worker.s(ws, j) - bst).to_bits(), Relaxed);
            }
        }
    }

    /// Regret matching+ keeps regrets non-negative; restore that after merging.
    pub(crate) fn clamp_regrets(&self) {
        for (slot, _) in self.occupied().collect::<Vec<_>>() {
            for j in 0..self.stride {
                if self.r(slot, j) < 0.0 {
                    self.set_r(slot, j, 0.0);
                }
            }
        }
    }

    fn occupied(&self) -> impl Iterator<Item = (usize, u64)> + '_ {
        (0..self.capacity()).filter_map(|i| match self.keys[i].load(Relaxed) {
            0 => None,
            t => Some((i, t - 1)),
        })
    }

    pub(crate) fn write(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&(self.len() as u64).to_le_bytes())?;
        for (i, key) in self.occupied() {
            w.write_all(&key.to_le_bytes())?;
            for j in 0..self.stride {
                w.write_all(&self.r(i, j).to_le_bytes())?;
            }
            for j in 0..self.stride {
                w.write_all(&self.s(i, j).to_le_bytes())?;
            }
        }
        Ok(())
    }

    pub(crate) fn read(&self, r: &mut impl Read) -> io::Result<()> {
        let count = read_u64(r)?;
        for _ in 0..count {
            let i = self.slot(read_u64(r)?);
            for j in 0..self.stride {
                self.set_r(i, j, f32::from_bits(read_u32(r)?));
            }
            for j in 0..self.stride {
                self.strat[i * self.stride + j].store(read_u64(r)?, Relaxed);
            }
        }
        Ok(())
    }

    /// Normalized average strategy for every info set with strategy mass.
    pub(crate) fn average(&self) -> FastMap<Vec<f32>> {
        let mut out = FastMap::default();
        for (i, key) in self.occupied() {
            let sum: f64 = (0..self.stride).map(|j| self.s(i, j)).sum();
            if sum > 0.0 {
                out.insert(key, (0..self.stride).map(|j| (self.s(i, j) / sum) as f32).collect());
            }
        }
        out
    }
}

pub struct Tables {
    pub call: Table,
    pub play: Table,
}

const CKPT_MAGIC: &[u8; 8] = b"LDCKPT06";
const POLICY_MAGIC: &[u8; 8] = b"LDPOL006";

impl Tables {
    /// Play slots are ~5x larger than call slots and fewer play states are reachable.
    pub fn new(log2_cap: u32) -> Tables {
        Tables { call: Table::new(log2_cap, CALL_STRIDE), play: Table::new(log2_cap - 1, PLAY_STRIDE) }
    }

    pub fn save_checkpoint(&self, path: &str, t: u64, n_players: usize) -> io::Result<()> {
        let tmp = format!("{path}.tmp");
        {
            let mut w = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
            w.write_all(CKPT_MAGIC)?;
            w.write_all(&t.to_le_bytes())?;
            w.write_all(&(n_players as u64).to_le_bytes())?;
            self.call.write(&mut w)?;
            self.play.write(&mut w)?;
            w.flush()?;
        }
        std::fs::rename(&tmp, path)
    }

    /// Returns (iterations, n_players).
    pub fn load_checkpoint(&self, path: &str) -> io::Result<(u64, usize)> {
        let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != CKPT_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a checkpoint (or info-set layout changed)"));
        }
        let t = read_u64(&mut r)?;
        let n = read_u64(&mut r)? as usize;
        self.call.read(&mut r)?;
        self.play.read(&mut r)?;
        Ok((t, n))
    }
}

/// Minimal hasher for u64 info-set keys (std's SipHash is needlessly slow here).
#[derive(Default)]
pub struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = hash64(self.0 ^ b as u64);
        }
    }
    fn write_u64(&mut self, k: u64) {
        self.0 = hash64(k);
    }
}

pub type FastMap<V> = HashMap<u64, V, BuildHasherDefault<KeyHasher>>;

/// Average strategy for every trained info set.
pub struct Policy {
    pub call: FastMap<Vec<f32>>,
    pub play: FastMap<Vec<f32>>,
    pub iterations: u64,
    /// Whether the policy was trained with per-player history buckets (false = ablation).
    pub history: bool,
    /// Trained for Devil mode.
    pub devil: bool,
    /// Play decisions see my own caught-lying record.
    pub self_image: bool,
    pub coarse_calls: bool,
    pub claimer_calls: bool,
    pub call_accuracy: bool,
    pub after_next: bool,
    pub threat: bool,
    pub full_table: bool,
    pub fast_reads: bool,
    pub call_table: bool,
}

impl Policy {
    pub fn from_tables(t: &Tables, iterations: u64, f: &Feat, devil: bool) -> Policy {
        Policy {
            call: t.call.average(), play: t.play.average(), iterations, history: f.history, devil, self_image: f.self_image,
            coarse_calls: f.coarse_calls, claimer_calls: f.claimer_calls, call_accuracy: f.call_accuracy,
            after_next: f.after_next, threat: f.threat, full_table: f.full_table, fast_reads: f.fast_reads, call_table: f.call_table,
        }
    }

    pub fn feat(&self) -> Feat {
        Feat {
            history: self.history, self_image: self.self_image, coarse_calls: self.coarse_calls,
            claimer_calls: self.claimer_calls, call_accuracy: self.call_accuracy, after_next: self.after_next, threat: self.threat, full_table: self.full_table, fast_reads: self.fast_reads, call_table: self.call_table,
        }
    }

    #[inline]
    pub fn call_probs(&self, key: u64) -> Option<&[f32]> {
        self.call.get(&key).map(|v| v.as_slice())
    }

    #[inline]
    pub fn play_probs(&self, key: u64) -> Option<&[f32]> {
        self.play.get(&key).map(|v| v.as_slice())
    }

    pub fn trained_counts(&self) -> (usize, usize) {
        (self.call.len(), self.play.len())
    }

    /// Sparse file: magic, iterations, flags u64 (1 history, 2 devil, 4 self image), then for call and play: count, (key u64, stride x f32).
    /// Keys are mixed-radix encodings of call_fields / play_fields (CALL_DIMS / PLAY_DIMS).
    pub fn save(&self, path: &str) -> io::Result<()> {
        let tmp = format!("{path}.tmp");
        {
            let mut w = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
            w.write_all(POLICY_MAGIC)?;
            w.write_all(&self.iterations.to_le_bytes())?;
            // Bits 8/16/32 are the ablation switches turned *off* (so 0 means the full design); bit 64
            // turns on the player-after-next play feature, bit 128 the threat feature, bit 256 the full-table summary, bit 512 fast reads, bit 1024 the call-side table view.
            let flags = self.history as u64 | (self.devil as u64) << 1 | (self.self_image as u64) << 2
                | (self.coarse_calls as u64) << 3 | (!self.claimer_calls as u64) << 4 | (!self.call_accuracy as u64) << 5
                | (self.after_next as u64) << 6 | (self.threat as u64) << 7 | (self.full_table as u64) << 8 | (self.fast_reads as u64) << 9 | (self.call_table as u64) << 10;
            w.write_all(&flags.to_le_bytes())?;
            for map in [&self.call, &self.play] {
                let mut keys: Vec<&u64> = map.keys().collect();
                keys.sort_unstable();
                w.write_all(&(keys.len() as u64).to_le_bytes())?;
                for k in keys {
                    w.write_all(&k.to_le_bytes())?;
                    for p in &map[k] {
                        w.write_all(&p.to_le_bytes())?;
                    }
                }
            }
            w.flush()?;
        }
        std::fs::rename(&tmp, path)
    }

    pub fn load(path: &str) -> io::Result<Policy> {
        let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != POLICY_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a policy file (or info-set layout changed)"));
        }
        let iterations = read_u64(&mut r)?;
        let flags = read_u64(&mut r)?;
        let (history, devil, self_image) = (flags & 1 != 0, flags & 2 != 0, flags & 4 != 0);
        let (coarse_calls, claimer_calls, call_accuracy) = (flags & 8 != 0, flags & 16 == 0, flags & 32 == 0);
        let (after_next, threat, full_table) = (flags & 64 != 0, flags & 128 != 0, flags & 256 != 0);
        let (fast_reads, call_table) = (flags & 512 != 0, flags & 1024 != 0);
        let mut maps = [FastMap::default(), FastMap::default()];
        for (map, st) in maps.iter_mut().zip([CALL_STRIDE, PLAY_STRIDE]) {
            let count = read_u64(&mut r)?;
            for _ in 0..count {
                let k = read_u64(&mut r)?;
                let mut v = Vec::with_capacity(st);
                for _ in 0..st {
                    v.push(f32::from_bits(read_u32(&mut r)?));
                }
                map.insert(k, v);
            }
        }
        let [call, play] = maps;
        Ok(Policy { call, play, iterations, history, devil, self_image, coarse_calls, claimer_calls, call_accuracy, after_next, threat, full_table, fast_reads, call_table })
    }
}

pub fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

pub fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
