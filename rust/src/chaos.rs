//! Chaos mode of Liar's Deck, as implemented by the game (decoded from ChaosDeckGamePlayManager /
//! ChaosDeckGameplay in the game's own code).
//!
//! - 12 cards: 5 Kings (type 1), 5 Queens (type 2), the Chaos card (type 3), the Master (type 4).
//!   Table card K or Q. 3 cards each. One card per play.
//! - The Master always counts as honest. A play is a lie when it is neither the table card nor the
//!   Master.
//! - Calling Liar: if the revealed card is the Master, its thrower shoots a target of their choice;
//!   otherwise on a lie the caller is forced to shoot the liar; otherwise (true) the caller pulls
//!   their own revolver.
//! - Throwing the Chaos card triggers it at once: every alive player shoots a target of their choice,
//!   then play continues with the next player (the Chaos card cannot be called).
//! - Shooting fires the shooter's own revolver: a miss advances their chamber; the bullet kills the
//!   target and reloads the shooter's revolver. Targets: opponents still holding cards, or any
//!   alive opponent if none do. Everyone dying at once is a draw (a loss for everyone here).
//! - After a call, the caller takes the next turn (next alive player if they died). Hands carry over
//!   with a new table card; a fresh deal only happens once every alive player is out of cards.
//! - The only player still holding cards must call.
//!
//! Decisions: call or pass, which card to play, and who to shoot. Each has its own table.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::agents::hypergeom_at_least;
use crate::engine::{death_prob, next_alive_of, CHAMBERS, MAX_PLAYERS};
use crate::policy::{read_u64, FastMap, Table};
use crate::rng::Rng;
use crate::value::{StartSampler, ValueModel};

/// Card kinds (absolute). The game's card types are these plus one.
pub const KING: usize = 0;
#[allow(dead_code)]
pub const QUEEN: usize = 1;
pub const CHAOS: usize = 2;
pub const MASTER: usize = 3;
const DECK: [u8; 4] = [5, 5, 1, 1];
const DECK_SIZE: u8 = 12;
const HAND: u8 = 3;

/// Why a free-choice shot is being fired (part of the target info set).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Shot {
    Master,
    Chaos,
}

#[derive(Clone)]
pub struct CGame {
    pub n: usize,
    pub alive: [bool; MAX_PLAYERS],
    pub shots: [u8; MAX_PLAYERS],
    /// Table rank this round: KING or QUEEN.
    pub table: usize,
    /// Cards in hand by absolute kind (private to each player).
    pub hand: [[u8; 4]; MAX_PLAYERS],
    /// Cards each player has played since the last full deal, by kind (private to each player).
    pub played: [[u8; 4]; MAX_PLAYERS],
    /// The Chaos card has been thrown (it is public) since the last full deal.
    pub chaos_out: bool,
    pub claims: [u8; MAX_PLAYERS],
    pub claims_total: u8,
    /// Last callable play: player and the kind of card (hidden until called).
    pub last: Option<(usize, usize)>,
    pub current: usize,
    pub starter: usize,
    pub round_over: bool,
    pub accused: usize,
    pub caller: usize,
    pub revealed: usize,
    /// Pending shots: who fires and at whom (a self-target is Russian roulette).
    pub target: [Option<usize>; MAX_PLAYERS],
    /// Who still has to pick a target for the pending free-choice shots.
    pub choosing: [bool; MAX_PLAYERS],
    pub game_over: bool,
    pub winner: Option<usize>,
    pub rec_called: [u8; MAX_PLAYERS],
    pub rec_lies: [u8; MAX_PLAYERS],
    pub rec_opps: [u8; MAX_PLAYERS],
    pub rec_calls: [u8; MAX_PLAYERS],
}

impl CGame {
    pub fn new(n: usize, alive: [bool; MAX_PLAYERS], shots: [u8; MAX_PLAYERS], starter: usize, rng: &mut Rng) -> CGame {
        let mut g = CGame {
            n,
            alive,
            shots,
            table: KING,
            hand: [[0; 4]; MAX_PLAYERS],
            played: [[0; 4]; MAX_PLAYERS],
            chaos_out: false,
            claims: [0; MAX_PLAYERS],
            claims_total: 0,
            last: None,
            current: starter,
            starter,
            round_over: false,
            accused: 0,
            caller: 0,
            revealed: KING,
            target: [None; MAX_PLAYERS],
            choosing: [false; MAX_PLAYERS],
            game_over: false,
            winner: None,
            rec_called: [0; MAX_PLAYERS],
            rec_lies: [0; MAX_PLAYERS],
            rec_opps: [0; MAX_PLAYERS],
            rec_calls: [0; MAX_PLAYERS],
        };
        g.deal(rng);
        g.start_round(starter, rng);
        g
    }

    pub fn new_game(n: usize, rng: &mut Rng) -> CGame {
        let mut alive = [false; MAX_PLAYERS];
        alive[..n].iter_mut().for_each(|a| *a = true);
        let starter = rng.below(n as u32) as usize;
        CGame::new(n, alive, [0; MAX_PLAYERS], starter, rng)
    }

    /// Full deal: fresh deck, 3 cards to every alive player.
    fn deal(&mut self, rng: &mut Rng) {
        let mut left = DECK;
        let mut total = DECK_SIZE as u32;
        self.chaos_out = false;
        for i in 0..self.n {
            self.hand[i] = [0; 4];
            self.played[i] = [0; 4];
            if !self.alive[i] {
                continue;
            }
            for _ in 0..HAND {
                let mut r = rng.below(total);
                for k in 0..4 {
                    if r < left[k] as u32 {
                        self.hand[i][k] += 1;
                        left[k] -= 1;
                        break;
                    }
                    r -= left[k] as u32;
                }
                total -= 1;
            }
        }
    }

    /// New round on the current hands: new table card, empty pile.
    fn start_round(&mut self, starter: usize, rng: &mut Rng) {
        self.table = rng.below(2) as usize;
        self.claims = [0; MAX_PLAYERS];
        self.claims_total = 0;
        self.last = None;
        self.current = starter;
        self.starter = starter;
        self.round_over = false;
        self.target = [None; MAX_PLAYERS];
        self.choosing = [false; MAX_PLAYERS];
        if !self.active(starter) {
            self.current = self.next_active(starter).unwrap_or(starter);
        }
    }

    #[inline]
    pub fn other(&self) -> usize {
        1 - self.table
    }

    #[inline]
    pub fn is_honest(&self, kind: usize) -> bool {
        kind == self.table || kind == MASTER
    }

    #[inline]
    pub fn hand_size(&self, p: usize) -> u8 {
        self.hand[p].iter().sum()
    }

    #[inline]
    pub fn n_alive(&self) -> usize {
        self.alive[..self.n].iter().filter(|&&a| a).count()
    }

    #[inline]
    fn active(&self, q: usize) -> bool {
        self.alive[q] && self.hand_size(q) > 0
    }

    pub fn next_active(&self, after: usize) -> Option<usize> {
        (1..self.n).map(|s| (after + s) % self.n).find(|&q| self.active(q))
    }

    pub fn others_active(&self, a: usize, b: usize) -> usize {
        (0..self.n).filter(|&q| q != a && q != b && self.active(q)).count()
    }

    pub fn is_forced(&self, p: usize) -> bool {
        self.last.is_some() && (self.hand_size(p) == 0 || !(0..self.n).any(|q| q != p && self.active(q)))
    }

    /// Card kinds `p` can play, in the order true rank, other rank, Master, Chaos (the play action
    /// index is the position here, so it means the same thing whichever rank is on the table).
    pub fn playable(&self, p: usize) -> ([usize; 4], usize) {
        let mut out = [0; 4];
        let mut k = 0;
        for kind in [self.table, self.other(), MASTER, CHAOS] {
            if self.hand[p][kind] > 0 {
                out[k] = kind;
                k += 1;
            }
        }
        (out, k)
    }

    /// Plays one card. The Chaos card triggers at once: every alive player must pick a target.
    pub fn apply_play(&mut self, kind: usize) {
        let p = self.current;
        debug_assert!(self.hand[p][kind] > 0 && !self.round_over);
        if self.last.is_some() {
            self.rec_opps[p] = self.rec_opps[p].saturating_add(1);
        }
        self.hand[p][kind] -= 1;
        self.played[p][kind] += 1;
        self.claims[p] += 1;
        self.claims_total += 1;
        if kind == CHAOS {
            self.chaos_out = true;
            self.last = None;
            self.accused = p;
            self.target = [None; MAX_PLAYERS];
            for q in 0..self.n {
                self.choosing[q] = self.alive[q];
            }
        } else {
            self.last = Some((p, kind));
        }
        self.current = self.next_active(p).unwrap_or_else(|| next_alive_of(&self.alive, self.n, p));
    }

    pub fn chaos_pending(&self) -> bool {
        !self.round_over && self.choosing.iter().any(|&c| c)
    }

    /// Resolves who has to fire. Free-choice targets are chosen afterwards (`choosing`).
    pub fn apply_call(&mut self) {
        let (a, kind) = self.last.expect("nothing to call");
        let c = self.current;
        if !self.is_forced(c) {
            self.rec_opps[c] = self.rec_opps[c].saturating_add(1);
            self.rec_calls[c] = self.rec_calls[c].saturating_add(1);
        }
        self.rec_called[a] = self.rec_called[a].saturating_add(1);
        let lie = !self.is_honest(kind);
        if lie {
            self.rec_lies[a] = self.rec_lies[a].saturating_add(1);
        }
        self.accused = a;
        self.caller = c;
        self.revealed = kind;
        self.round_over = true;
        self.target = [None; MAX_PLAYERS];
        self.choosing = [false; MAX_PLAYERS];
        if kind == MASTER {
            self.choosing[a] = true;
        } else if lie {
            self.target[c] = Some(a);
        } else {
            self.target[c] = Some(c);
        }
    }

    /// Players still picking a free-choice target, and why.
    pub fn choosers(&self) -> (Vec<usize>, Shot) {
        let why = if self.round_over { Shot::Master } else { Shot::Chaos };
        ((0..self.n).filter(|&q| self.choosing[q]).collect(), why)
    }

    /// Valid targets for shooter `s`: alive opponents still holding cards, or any alive opponent.
    pub fn valid_targets(&self, s: usize) -> [bool; MAX_PLAYERS] {
        let mut v = [false; MAX_PLAYERS];
        for q in 0..self.n {
            v[q] = q != s && self.alive[q] && self.hand_size(q) > 0;
        }
        if !v.iter().any(|&x| x) {
            for q in 0..self.n {
                v[q] = q != s && self.alive[q];
            }
        }
        v
    }

    pub fn set_target(&mut self, s: usize, t: usize) {
        self.target[s] = Some(t);
        self.choosing[s] = false;
    }

    /// Fires every pending shot at once.
    fn fire(&mut self, rng: &mut Rng) {
        let mut hit = [false; MAX_PLAYERS];
        let mut died = [false; MAX_PLAYERS];
        for s in 0..self.n {
            if let Some(t) = self.target[s] {
                if rng.below((CHAMBERS - self.shots[s]) as u32) == 0 {
                    hit[s] = true;
                    died[t] = true;
                }
            }
        }
        for q in 0..self.n {
            if died[q] {
                self.alive[q] = false;
            }
        }
        for s in 0..self.n {
            if self.target[s].is_some() && self.alive[s] {
                // The bullet reloads the shooter's revolver; a miss advances the chamber.
                self.shots[s] = if hit[s] { 0 } else { self.shots[s] + 1 };
            }
        }
        self.target = [None; MAX_PLAYERS];
        self.choosing = [false; MAX_PLAYERS];
        if self.n_alive() <= 1 {
            self.game_over = true;
            self.winner = (0..self.n).find(|&q| self.alive[q]);
        }
    }

    /// After a Chaos throw has all its targets: fire, then play continues.
    pub fn resolve_chaos(&mut self, rng: &mut Rng) {
        self.fire(rng);
        if self.game_over {
            return;
        }
        if !(0..self.n).any(|q| self.active(q)) {
            // The Chaos card was the last card in play: nothing left to call, deal afresh.
            let next = next_alive_of(&self.alive, self.n, self.accused);
            self.deal(rng);
            self.start_round(next, rng);
        } else if !self.active(self.current) {
            if let Some(q) = self.next_active(self.current) {
                self.current = q;
            }
        }
    }

    /// After a call has all its targets: fire, then start the next round (or deal afresh).
    pub fn finish_round(&mut self, rng: &mut Rng) {
        self.fire(rng);
        if self.game_over {
            return;
        }
        let starter = if self.alive[self.caller] { self.caller } else { next_alive_of(&self.alive, self.n, self.caller) };
        if !(0..self.n).any(|q| self.active(q)) {
            self.deal(rng);
        }
        self.start_round(starter, rng);
    }

    /// Expected value for `me` once the pending shots fire, every hit/miss combination weighed.
    /// `next` is who would act next (the caller after a call, the current player after Chaos).
    pub fn value_after_shots(&self, value: &ValueModel, me: usize, next: usize) -> f64 {
        if self.game_over {
            return if self.alive[me] && self.n_alive() == 1 { 1.0 } else { 0.0 };
        }
        let shooters: Vec<(usize, usize)> = (0..self.n).filter_map(|s| self.target[s].map(|t| (s, t))).collect();
        let mut total = 0.0;
        for mask in 0u32..(1 << shooters.len()) {
            let mut p = 1.0;
            let mut died = [false; MAX_PLAYERS];
            for (j, &(s, t)) in shooters.iter().enumerate() {
                let pd = death_prob(self.shots[s]);
                if mask >> j & 1 == 1 {
                    p *= pd;
                    died[t] = true;
                } else {
                    p *= 1.0 - pd;
                }
            }
            if p == 0.0 {
                continue;
            }
            let mut alive = self.alive;
            let mut shots = self.shots;
            for q in 0..self.n {
                if died[q] {
                    alive[q] = false;
                }
            }
            for (j, &(s, _)) in shooters.iter().enumerate() {
                if alive[s] {
                    shots[s] = if mask >> j & 1 == 1 { 0 } else { shots[s] + 1 };
                }
            }
            if !alive[me] {
                continue;
            }
            let starter = if alive[next] { next } else { next_alive_of(&alive, self.n, next) };
            total += p * value.state_value(&alive, &shots, starter, me);
        }
        total
    }
}

// ── Info sets ────────────────────────────────────────────────────────────────

fn mix(dims: &[u64], vals: &[u64]) -> u64 {
    let mut k = 0;
    for (d, v) in dims.iter().zip(vals) {
        debug_assert!(v < d, "chaos key field out of range: {v} >= {d}");
        k = k * d + v;
    }
    k
}

fn lie_bucket(g: &CGame, q: usize) -> u64 {
    let c = g.rec_called[q] as f32;
    if c < 3.0 {
        return 0;
    }
    let r = (g.rec_lies[q] as f32 + 1.0) / (c + 2.0);
    if r < 0.4 { 1 } else if r > 0.6 { 3 } else { 2 }
}

fn call_bucket(g: &CGame, q: usize) -> u64 {
    let o = g.rec_opps[q] as f32;
    if o < 5.0 {
        return 0;
    }
    let r = g.rec_calls[q] as f32 / o;
    if r < 0.15 { 1 } else if r > 0.4 { 3 } else { 2 }
}

/// Honest cards (table rank + Master) I can't account for since the last deal.
fn unseen_honest(g: &CGame, p: usize) -> u8 {
    let t = g.table;
    let rank = DECK[t] - g.hand[p][t] - g.played[p][t];
    let master = DECK[MASTER] - g.hand[p][MASTER] - g.played[p][MASTER];
    rank + master
}

/// My hand relative to the table: true rank, other rank, Master, Chaos.
fn hand_fields(g: &CGame, p: usize) -> [u64; 4] {
    let h = g.hand[p];
    [h[g.table] as u64, h[g.other()] as u64, h[MASTER] as u64, h[CHAOS] as u64]
}

/// Could the Chaos card still be thrown by someone else (0 no, 1 yes)?
fn chaos_elsewhere(g: &CGame, p: usize) -> u64 {
    (!g.chaos_out && g.hand[p][CHAOS] == 0 && g.played[p][CHAOS] == 0) as u64
}

// n_alive, my hand (true / other / Master / Chaos), claimer total claims, third-party claims,
// claimer cards left, my pulls, claimer pulls, others with cards, unseen honest, claimer lie record,
// Chaos still out elsewhere.
const CALL_DIMS: [u64; 14] = [3, 4, 4, 2, 2, 3, 6, 3, 6, 6, 3, 7, 4, 2];

pub fn call_key(g: &CGame, p: usize) -> u64 {
    let (prev, _) = g.last.expect("call key without a play");
    let h = hand_fields(g, p);
    let third = g.claims_total - g.claims[p] - g.claims[prev];
    mix(&CALL_DIMS, &[
        g.n_alive() as u64 - 2, h[0], h[1], h[2], h[3],
        (g.claims[prev] as u64).clamp(1, 3) - 1,
        (third as u64).min(5),
        g.hand_size(prev).min(2) as u64,
        g.shots[p] as u64, g.shots[prev] as u64,
        g.others_active(p, prev).min(2) as u64,
        unseen_honest(g, p).min(6) as u64,
        lie_bucket(g, prev),
        chaos_elsewhere(g, p),
    ])
}

// n_alive, my hand, next player's cards, my pulls, next pulls, round claims, unseen honest,
// others with cards, next player's calling record, Chaos still out elsewhere, opening the pile.
const PLAY_DIMS: [u64; 15] = [3, 4, 4, 2, 2, 4, 6, 6, 7, 7, 3, 4, 2, 2, 1];

pub fn play_key(g: &CGame, p: usize) -> u64 {
    let nxt = g.next_active(p);
    let h = hand_fields(g, p);
    let (nl, ns) = nxt.map_or((0, 0), |q| (g.hand_size(q) as u64, g.shots[q] as u64));
    mix(&PLAY_DIMS, &[
        g.n_alive() as u64 - 2, h[0], h[1], h[2], h[3], nl.min(3), g.shots[p] as u64, ns,
        (g.claims_total as u64).min(6), unseen_honest(g, p).min(6) as u64,
        g.others_active(p, nxt.unwrap_or(p)).min(2) as u64,
        nxt.map_or(0, |q| call_bucket(g, q)),
        chaos_elsewhere(g, p),
        g.last.is_none() as u64,
        0,
    ])
}

/// Targets as relative seats 1..=3 after the shooter: 0 = not a valid target, else
/// 1 + pulls * 4 + cards in hand (so the policy sees how dangerous each candidate is).
fn target_slots(g: &CGame, s: usize) -> [u64; 3] {
    let v = g.valid_targets(s);
    let mut out = [0u64; 3];
    for (r, o) in out.iter_mut().enumerate() {
        if r + 1 < g.n {
            let q = (s + r + 1) % g.n;
            if v[q] {
                *o = 1 + g.shots[q] as u64 * 4 + g.hand_size(q).min(3) as u64;
            }
        }
    }
    out
}

const TARGET_DIMS: [u64; 6] = [3, 2, 6, 25, 25, 25];

pub fn target_key(g: &CGame, s: usize, why: Shot) -> u64 {
    let t = target_slots(g, s);
    let w = match why {
        Shot::Master => 0,
        Shot::Chaos => 1,
    };
    mix(&TARGET_DIMS, &[g.n_alive() as u64 - 2, w, g.shots[s] as u64, t[0], t[1], t[2]])
}

/// Valid targets of `s` in relative-seat order (the target action index is the position here).
pub fn target_choices(g: &CGame, s: usize) -> Vec<usize> {
    let v = g.valid_targets(s);
    (1..g.n).map(|r| (s + r) % g.n).filter(|&q| v[q]).collect()
}

// ── Agents ───────────────────────────────────────────────────────────────────

pub trait CAgent {
    fn call(&mut self, g: &CGame, p: usize, rng: &mut Rng) -> bool;
    /// Index into `g.playable(p)`.
    fn play(&mut self, g: &CGame, p: usize, rng: &mut Rng) -> usize;
    /// Index into `target_choices(g, s)`.
    fn target(&mut self, g: &CGame, s: usize, why: Shot, rng: &mut Rng) -> usize;
}

fn index_of(g: &CGame, p: usize, kind: usize) -> usize {
    let (ks, n) = g.playable(p);
    (0..n).find(|&i| ks[i] == kind).unwrap_or(0)
}

/// The most dangerous valid target: fewest pulls taken (likeliest to outlast us), then most cards.
fn strongest_target(g: &CGame, s: usize) -> usize {
    let ch = target_choices(g, s);
    (0..ch.len()).min_by_key(|&i| (g.shots[ch[i]], 3 - g.hand_size(ch[i]).min(3))).unwrap_or(0)
}

pub struct CRandom;

impl CAgent for CRandom {
    fn call(&mut self, _: &CGame, _: usize, rng: &mut Rng) -> bool {
        rng.f32() < 0.3
    }
    fn play(&mut self, g: &CGame, p: usize, rng: &mut Rng) -> usize {
        rng.below(g.playable(p).1 as u32) as usize
    }
    fn target(&mut self, g: &CGame, s: usize, _: Shot, rng: &mut Rng) -> usize {
        rng.below(target_choices(g, s).len().max(1) as u32) as usize
    }
}

/// Plays honest cards first, then the other rank, then Chaos; calls only on impossible claims.
pub struct CHonest;

impl CAgent for CHonest {
    fn call(&mut self, g: &CGame, p: usize, _: &mut Rng) -> bool {
        let (prev, _) = g.last.unwrap();
        g.claims[prev] > unseen_honest(g, p)
    }
    fn play(&mut self, g: &CGame, p: usize, _: &mut Rng) -> usize {
        for kind in [g.table, MASTER, g.other(), CHAOS] {
            if g.hand[p][kind] > 0 {
                return index_of(g, p, kind);
            }
        }
        0
    }
    fn target(&mut self, g: &CGame, s: usize, _: Shot, _: &mut Rng) -> usize {
        strongest_target(g, s)
    }
}

/// Card-count caller (a lie makes it shoot the liar, a truth costs it a pull), bluffs sometimes,
/// throws the Chaos card now and then, shoots the strongest opponent.
pub struct CProb {
    pub threshold: f64,
    pub bluff: f32,
    pub special: f32,
}

impl Default for CProb {
    fn default() -> Self {
        CProb { threshold: 0.45, bluff: 0.2, special: 0.25 }
    }
}

impl CAgent for CProb {
    fn call(&mut self, g: &CGame, p: usize, _: &mut Rng) -> bool {
        let (prev, _) = g.last.unwrap();
        let unknown = (DECK_SIZE - g.hand_size(p)) as u64;
        let p_true = hypergeom_at_least(g.claims[prev] as u64, unknown, unseen_honest(g, p) as u64, HAND as u64);
        let thr = (self.threshold + 0.5 * (death_prob(g.shots[p]) - 1.0 / 6.0)).clamp(0.05, 0.95);
        1.0 - p_true > thr
    }
    fn play(&mut self, g: &CGame, p: usize, rng: &mut Rng) -> usize {
        let h = g.hand[p];
        if h[CHAOS] > 0 && rng.f32() < self.special {
            return index_of(g, p, CHAOS);
        }
        let honest = if h[g.table] > 0 { Some(g.table) } else if h[MASTER] > 0 { Some(MASTER) } else { None };
        if let Some(k) = honest {
            if !(h[g.other()] > 0 && rng.f32() < self.bluff) {
                return index_of(g, p, k);
            }
        }
        for kind in [g.other(), g.table, MASTER, CHAOS] {
            if h[kind] > 0 {
                return index_of(g, p, kind);
            }
        }
        0
    }
    fn target(&mut self, g: &CGame, s: usize, _: Shot, _: &mut Rng) -> usize {
        strongest_target(g, s)
    }
}

pub struct CPolicyAgent {
    pub policy: Arc<ChaosPolicy>,
    pub fallback: CProb,
}

impl CPolicyAgent {
    pub fn new(policy: Arc<ChaosPolicy>) -> Self {
        CPolicyAgent { policy, fallback: CProb::default() }
    }
}

impl CAgent for CPolicyAgent {
    fn call(&mut self, g: &CGame, p: usize, rng: &mut Rng) -> bool {
        match self.policy.tables[0].get(&call_key(g, p)) {
            Some(pr) => rng.sample(pr, 2) == 1,
            None => self.fallback.call(g, p, rng),
        }
    }
    fn play(&mut self, g: &CGame, p: usize, rng: &mut Rng) -> usize {
        match self.policy.tables[1].get(&play_key(g, p)) {
            Some(pr) => rng.sample(pr, g.playable(p).1),
            None => self.fallback.play(g, p, rng),
        }
    }
    fn target(&mut self, g: &CGame, s: usize, why: Shot, rng: &mut Rng) -> usize {
        let n = target_choices(g, s).len();
        match self.policy.tables[2].get(&target_key(g, s, why)) {
            Some(pr) if n > 0 => rng.sample(pr, n),
            _ => self.fallback.target(g, s, why, rng),
        }
    }
}

pub fn make_agent(kind: &str, policy: Option<&Arc<ChaosPolicy>>) -> Box<dyn CAgent> {
    match kind {
        "prob" => Box::new(CProb::default()),
        "honest" => Box::new(CHonest),
        "random" => Box::new(CRandom),
        "aggro" => Box::new(CProb { threshold: 0.3, bluff: 0.5, special: 0.6 }),
        "timid" => Box::new(CProb { threshold: 0.7, bluff: 0.05, special: 0.05 }),
        _ => Box::new(CPolicyAgent::new(policy.expect("policy required").clone())),
    }
}

/// Lets every pending chooser pick a target.
fn choose_targets(g: &mut CGame, agents: &mut [Box<dyn CAgent>], rng: &mut Rng) {
    let (choosers, why) = g.choosers();
    for s in choosers {
        let ch = target_choices(g, s);
        if ch.is_empty() {
            g.choosing[s] = false;
            continue;
        }
        let i = agents[s].target(g, s, why, rng).min(ch.len() - 1);
        g.set_target(s, ch[i]);
    }
}

/// Plays one full game; returns the winner (None for a draw).
pub fn play_game(agents: &mut [Box<dyn CAgent>], rng: &mut Rng) -> Option<usize> {
    let mut g = CGame::new_game(agents.len(), rng);
    let mut steps = 0;
    while !g.game_over {
        steps += 1;
        assert!(steps < 100_000, "chaos game exceeded step limit");
        let p = g.current;
        if g.is_forced(p) || (g.last.is_some() && agents[p].call(&g, p, rng)) {
            g.apply_call();
        } else {
            let (ks, _) = g.playable(p);
            let i = agents[p].play(&g, p, rng);
            g.apply_play(ks[i]);
            if g.chaos_pending() {
                choose_targets(&mut g, agents, rng);
                g.resolve_chaos(rng);
                continue;
            }
        }
        if g.round_over {
            choose_targets(&mut g, agents, rng);
            g.finish_round(rng);
        }
    }
    g.winner
}

pub fn evaluate(policy: &Arc<ChaosPolicy>, opponents: &str, games: usize, n: usize, seed: u64) -> f64 {
    let mut rng = Rng::new(seed);
    let lineup: Vec<&str> = match opponents {
        "mixed" => (0..n - 1).map(|i| ["prob", "honest", "random"][i % 3]).collect(),
        "newmix" => (0..n - 1).map(|i| ["aggro", "timid", "honest"][i % 3]).collect(),
        k => vec![k; n - 1],
    };
    let mut wins = 0;
    for gi in 0..games {
        let seat = gi % n;
        let mut agents: Vec<Box<dyn CAgent>> = Vec::with_capacity(n);
        let mut it = lineup.iter();
        for s in 0..n {
            agents.push(if s == seat { make_agent("policy", Some(policy)) } else { make_agent(it.next().unwrap(), Some(policy)) });
        }
        if play_game(&mut agents, &mut rng) == Some(seat) {
            wins += 1;
        }
    }
    wins as f64 / games as f64
}

// ── Policy and tables ────────────────────────────────────────────────────────

const STRIDES: [usize; 3] = [2, 4, 3];
const CKPT_MAGIC: &[u8; 8] = b"LDCHCK02";
const POLICY_MAGIC: &[u8; 8] = b"LDCHPL02";

pub struct ChaosTables {
    /// Call, play, target.
    pub t: [Table; 3],
}

impl ChaosTables {
    pub fn new(log2: u32) -> Self {
        ChaosTables { t: [Table::new(log2, STRIDES[0]), Table::new(log2 - 1, STRIDES[1]), Table::new(log2 - 2, STRIDES[2])] }
    }

    pub fn save(&self, path: &str, iterations: u64, n: usize) -> io::Result<()> {
        let tmp = format!("{path}.tmp");
        {
            let mut w = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
            w.write_all(CKPT_MAGIC)?;
            w.write_all(&iterations.to_le_bytes())?;
            w.write_all(&(n as u64).to_le_bytes())?;
            for t in &self.t {
                t.write(&mut w)?;
            }
            w.flush()?;
        }
        std::fs::rename(&tmp, path)
    }

    pub fn load(&self, path: &str) -> io::Result<(u64, usize)> {
        let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != CKPT_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a Chaos checkpoint (or info-set layout changed)"));
        }
        let it = read_u64(&mut r)?;
        let n = read_u64(&mut r)? as usize;
        for t in &self.t {
            t.read(&mut r)?;
        }
        Ok((it, n))
    }
}

/// Average strategy for Chaos: call, play and target tables.
pub struct ChaosPolicy {
    pub tables: [FastMap<Vec<f32>>; 3],
    pub iterations: u64,
}

impl ChaosPolicy {
    pub fn from_tables(t: &ChaosTables, iterations: u64) -> Self {
        ChaosPolicy { tables: [t.t[0].average(), t.t[1].average(), t.t[2].average()], iterations }
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        let tmp = format!("{path}.tmp");
        {
            let mut w = BufWriter::with_capacity(1 << 20, File::create(&tmp)?);
            w.write_all(POLICY_MAGIC)?;
            w.write_all(&self.iterations.to_le_bytes())?;
            for map in &self.tables {
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

    pub fn load(path: &str) -> io::Result<Self> {
        let mut r = BufReader::with_capacity(1 << 20, File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != POLICY_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a Chaos policy file (or info-set layout changed)"));
        }
        let iterations = read_u64(&mut r)?;
        let mut tables = [FastMap::default(), FastMap::default(), FastMap::default()];
        for (map, st) in tables.iter_mut().zip(STRIDES) {
            let count = read_u64(&mut r)?;
            for _ in 0..count {
                let k = read_u64(&mut r)?;
                let mut v = Vec::with_capacity(st);
                for _ in 0..st {
                    let mut b = [0u8; 4];
                    r.read_exact(&mut b)?;
                    v.push(f32::from_le_bytes(b));
                }
                map.insert(k, v);
            }
        }
        Ok(ChaosPolicy { tables, iterations })
    }
}

// ── Trainer ──────────────────────────────────────────────────────────────────

const MAX_A: usize = 4;

pub struct Rec {
    table: u8,
    slot: u32,
    n: u8,
    a: u8,
    sig: [f32; MAX_A],
}

/// Outcome-sampling MCCFR over full Chaos games (same scheme as the deck trainer): each round is
/// one iteration traversed by a random alive policy seat, paid its change in win probability.
/// Chaos throws resolve mid-round by sampling; a call's shots are valued exactly.
pub struct ChaosTrainer {
    pub tables: ChaosTables,
    pub t: AtomicU64,
    pub n: usize,
    pub epsilon: f32,
    pub bot_share: f32,
    pub value: ValueModel,
    pub starts: StartSampler,
}

impl ChaosTrainer {
    #[allow(clippy::too_many_arguments)]
    fn act(&self, id: usize, key: u64, n: usize, trav: bool, t: u64, rng: &mut Rng, recs: &mut Vec<Rec>) -> (usize, f64) {
        let tbl = &self.tables.t[id];
        let slot = tbl.slot(key);
        let mut sig10 = [0.0f32; crate::engine::MAX_ACTIONS];
        tbl.sigma(slot, n, &mut sig10);
        let mut sig = [0.0f32; MAX_A];
        sig[..n].copy_from_slice(&sig10[..n]);
        if trav {
            let mut probs = [0.0f32; MAX_A];
            for k in 0..n {
                probs[k] = self.epsilon / n as f32 + (1.0 - self.epsilon) * sig[k];
            }
            let a = rng.sample(&probs, n);
            recs.push(Rec { table: id as u8, slot: slot as u32, n: n as u8, a: a as u8, sig });
            (a, probs[a] as f64)
        } else {
            let a = rng.sample(&sig, n);
            tbl.add_strat(slot, n, &sig10, t as f64);
            (a, 1.0)
        }
    }

    fn update(&self, recs: &[Rec], u: f64, q: f64) {
        if u == 0.0 || recs.is_empty() {
            return;
        }
        let mut tail = 1.0f64;
        for rec in recs.iter().rev() {
            let w = (u * tail / q) as f32;
            let tbl = &self.tables.t[rec.table as usize];
            let (slot, a) = (rec.slot as usize, rec.a as usize);
            let sa = rec.sig[a];
            for b in 0..rec.n as usize {
                let r = tbl.r(slot, b);
                let v = if b == a { r + w * (1.0 - sa) } else { r - w * sa };
                tbl.set_r(slot, b, if v > 0.0 { v } else { 0.0 });
            }
            tail *= sa as f64;
        }
    }

    pub fn iterate(&self, rng: &mut Rng, recs: &mut Vec<Rec>, bots: &mut [Box<dyn CAgent>; 3]) {
        let n = self.n;
        let (alive, shots, starter) = if rng.f32() < 0.5 {
            let mut a = [false; MAX_PLAYERS];
            a[..n].iter_mut().for_each(|x| *x = true);
            (a, [0u8; MAX_PLAYERS], rng.below(n as u32) as usize)
        } else {
            self.starts.sample(rng)
        };
        let mut g = CGame::new(n, alive, shots, starter, rng);
        let mut seat = [0u8; MAX_PLAYERS];
        for k in seat.iter_mut().take(n) {
            if rng.f32() < self.bot_share {
                *k = 1 + rng.below(3) as u8;
            }
        }
        let pick = rng.below(g.n_alive() as u32) as usize;
        seat[(0..n).filter(|&q| alive[q]).nth(pick).unwrap()] = 0;
        loop {
            let mine: Vec<usize> = (0..n).filter(|&q| g.alive[q] && seat[q] == 0).collect();
            if mine.is_empty() {
                return;
            }
            let trav = mine[rng.below(mine.len() as u32) as usize];
            let t = self.t.fetch_add(1, Relaxed) + 1;
            self.play_round(&mut g, &seat, trav, t, rng, recs, bots);
            if g.game_over {
                return;
            }
            g.finish_round(rng);
            if g.game_over {
                return;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn pick_targets(&self, g: &mut CGame, seat: &[u8; MAX_PLAYERS], trav: usize, t: u64, rng: &mut Rng, recs: &mut Vec<Rec>, bots: &mut [Box<dyn CAgent>; 3], q: &mut f64) {
        let (choosers, why) = g.choosers();
        for sh in choosers {
            let ch = target_choices(g, sh);
            if ch.is_empty() {
                g.choosing[sh] = false;
                continue;
            }
            let i = if seat[sh] != 0 {
                bots[seat[sh] as usize - 1].target(g, sh, why, rng).min(ch.len() - 1)
            } else if ch.len() == 1 {
                0
            } else {
                let (a, s) = self.act(2, target_key(g, sh, why), ch.len(), sh == trav, t, rng, recs);
                *q *= s;
                a
            };
            g.set_target(sh, ch[i]);
        }
    }

    /// Plays one round up to its call (or the end of the game) and updates the traverser.
    #[allow(clippy::too_many_arguments)]
    fn play_round(&self, g: &mut CGame, seat: &[u8; MAX_PLAYERS], trav: usize, t: u64, rng: &mut Rng, recs: &mut Vec<Rec>, bots: &mut [Box<dyn CAgent>; 3]) {
        let baseline = self.value.state_value(&g.alive, &g.shots, g.starter, trav);
        recs.clear();
        let mut q = 1.0f64;
        while !g.round_over && !g.game_over {
            let p = g.current;
            if g.is_forced(p) {
                g.apply_call();
                break;
            }
            if seat[p] != 0 {
                let bot = &mut bots[seat[p] as usize - 1];
                if g.last.is_some() && bot.call(g, p, rng) {
                    g.apply_call();
                    break;
                }
                let (ks, _) = g.playable(p);
                let i = bot.play(g, p, rng);
                g.apply_play(ks[i]);
            } else {
                if g.last.is_some() {
                    let (a, s) = self.act(0, call_key(g, p), 2, p == trav, t, rng, recs);
                    q *= s;
                    if a == 1 {
                        g.apply_call();
                        break;
                    }
                }
                let (ks, nk) = g.playable(p);
                let (a, s) = self.act(1, play_key(g, p), nk, p == trav, t, rng, recs);
                q *= s;
                g.apply_play(ks[a]);
            }
            if g.chaos_pending() {
                self.pick_targets(g, seat, trav, t, rng, recs, bots, &mut q);
                g.resolve_chaos(rng);
            }
        }
        if g.round_over {
            self.pick_targets(g, seat, trav, t, rng, recs, bots, &mut q);
        }
        let v = g.value_after_shots(&self.value, trav, g.caller);
        self.update(recs, v - baseline, q);
    }
}

pub struct TrainOpts {
    pub n: usize,
    pub iterations: u64,
    pub threads: usize,
    pub eval_every: u64,
    pub eval_games: usize,
    pub seed: u64,
    pub checkpoint: String,
    pub out: String,
    pub table_log: u32,
    pub epsilon: f32,
    pub bot_share: f32,
    pub resume: bool,
}

pub fn train(o: TrainOpts) {
    let tables = ChaosTables::new(o.table_log);
    let mut t0 = 0;
    if o.resume && std::path::Path::new(&o.checkpoint).exists() {
        let (t, n) = tables.load(&o.checkpoint).expect("loading Chaos checkpoint");
        assert_eq!(n, o.n, "checkpoint was trained with {n} players");
        t0 = t;
        println!("resumed from {} at {t}", o.checkpoint);
    }
    let tr = ChaosTrainer {
        tables,
        t: AtomicU64::new(t0),
        n: o.n,
        epsilon: o.epsilon,
        bot_share: o.bot_share,
        value: ValueModel::new(o.n),
        starts: StartSampler::prior(o.n),
    };
    let target = t0 + o.iterations;
    println!("Chaos: training to {target} rounds on {} threads (bot share {})", o.threads, o.bot_share);
    let start = Instant::now();
    let mut chunk = 0u64;
    while tr.t.load(Relaxed) < target {
        let cur = tr.t.load(Relaxed);
        let end = ((cur / o.eval_every + 1) * o.eval_every).min(target);
        chunk += 1;
        std::thread::scope(|s| {
            for th in 0..o.threads {
                let tr = &tr;
                s.spawn(move || {
                    let mut rng = Rng::new(o.seed.wrapping_mul(1_000_003) ^ (chunk << 20) ^ th as u64 ^ cur ^ 0xC4A05);
                    let mut recs = Vec::with_capacity(32);
                    let mut bots: [Box<dyn CAgent>; 3] = [Box::new(CHonest), Box::new(CProb::default()), Box::new(CRandom)];
                    while tr.t.load(Relaxed) < end {
                        for _ in 0..256 {
                            tr.iterate(&mut rng, &mut recs, &mut bots);
                        }
                    }
                });
            }
            while tr.t.load(Relaxed) < end {
                std::thread::sleep(Duration::from_millis(200));
            }
        });
        let t = tr.t.load(Relaxed);
        tr.tables.save(&o.checkpoint, t, o.n).expect("saving checkpoint");
        let policy = Arc::new(ChaosPolicy::from_tables(&tr.tables, t));
        policy.save(&o.out).expect("saving policy");
        let parts: Vec<String> = ["prob", "mixed", "newmix"]
            .iter()
            .map(|opp| format!("vs {opp}: {:.3}", evaluate(&policy, opp, o.eval_games, o.n, t)))
            .collect();
        println!("eval @ {t} | {:.0}s | info sets {} call {} play {} target | baseline {:.3} | {}",
            start.elapsed().as_secs_f64(), policy.tables[0].len(), policy.tables[1].len(), policy.tables[2].len(),
            1.0 / o.n as f64, parts.join(" | "));
    }
}

pub fn eval_all(policy: &Arc<ChaosPolicy>, n: usize, games: usize, seed: u64) {
    println!("Chaos policy trained for {} rounds | players {n} | games {games} | baseline {:.3}", policy.iterations, 1.0 / n as f64);
    for opp in ["prob", "honest", "random", "mixed", "aggro", "timid", "newmix", "policy"] {
        println!("vs {opp:<7} win rate {:.3}", evaluate(policy, opp, games, n, seed));
    }
}
