use crate::rng::Rng;

pub const MAX_PLAYERS: usize = 4;
pub const CHAMBERS: u8 = 6;
pub const DECK_SIZE: u8 = 20;
pub const MATCHING_IN_DECK: u8 = 8; // 6 of the table rank + 2 Jokers
pub const HAND_SIZE: u8 = 5;
/// 9 ordinary plays plus playing the Devil card alone.
pub const MAX_ACTIONS: usize = 10;

/// Who opens the next round.
///   Loser: the player who pulled the trigger, or the next alive player if they died.
///   Random: a random alive player.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StarterRule {
    Loser,
    Random,
}

impl StarterRule {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "loser" => Some(StarterRule::Loser),
            "random" => Some(StarterRule::Random),
            _ => None,
        }
    }
}

#[inline]
pub fn death_prob(shots: u8) -> f64 {
    1.0 / (CHAMBERS - shots) as f64
}

/// Legal plays for a hand with `m` matching and `nm` non-matching cards, as
/// (true cards, bluff cards), ordered by total size then most-true first.
#[derive(Clone, Copy)]
pub struct Legal {
    pub n: usize,
    pub acts: [(u8, u8); MAX_ACTIONS],
}

const fn build_legal() -> [[Legal; 6]; 6] {
    let mut out = [[Legal { n: 0, acts: [(0, 0); MAX_ACTIONS] }; 6]; 6];
    let mut m = 0;
    while m < 6 {
        let mut nm = 0;
        while nm < 6 {
            let mut acts = [(0u8, 0u8); MAX_ACTIONS];
            let mut k = 0;
            let mut total = 1i32;
            while total <= 3 {
                let mut t = total;
                while t >= 0 {
                    let b = total - t;
                    if t <= m as i32 && b <= nm as i32 {
                        acts[k] = (t as u8, b as u8);
                        k += 1;
                    }
                    t -= 1;
                }
                total += 1;
            }
            out[m][nm] = Legal { n: k, acts };
            nm += 1;
        }
        m += 1;
    }
    out
}

pub static LEGAL: [[Legal; 6]; 6] = build_legal();

#[derive(Clone, Copy, Debug)]
pub struct Play {
    pub player: usize,
    pub n: u8,
    pub n_true: u8,
    /// The Devil card played alone (hidden until called).
    pub devil: bool,
}

/// One public event, for the history the transformer network reads. Seats are absolute.
/// kind: 1 claim, 2 Liar call, 3 Devil's Deal, 4 pull, 5 round start.
/// flag: call 0 truth / 1 lie / 2 Devil revealed; deal 1 won; pull 1 died.
#[derive(Clone, Copy, Debug, Default)]
pub struct Ev {
    pub kind: u8,
    pub actor: u8,
    pub target: u8,
    pub n: u8,
    pub left: u8,
    pub total: u8,
    pub flag: u8,
    /// Claims only: times the claimer checked their cards while waiting (capped), and how long
    /// they took relative to their usual pace (log2 ratio, in 1/32 steps). Simulated players get
    /// them from rates fitted to the logged lobbies (`tells`); the advisor fills in real ones.
    pub looks: u8,
    pub pace: i8,
    /// Claims only: the claimer's usual card-check rate from earlier games (share of claims
    /// with a check, x255, shrunk toward the lobby average) and how many claims it rests on
    /// (capped at 30; 0 = a stranger).
    pub habit: u8,
    pub habit_n: u8,
}

/// Events kept per game (the network reads the most recent ones).
pub const HIST_KEEP: usize = 64;

/// `decide_play` result meaning "play the Devil card alone".
pub const DEVIL_PLAY: (u8, u8) = (0, 0);

/// A mixed play (real + fake) is a lie when called, exactly like a pure bluff of the same size,
/// and spends a real card. When that pure bluff is possible, play it instead.
#[inline]
pub fn undominate(hn: u8, a: (u8, u8)) -> (u8, u8) {
    if a.0 > 0 && a.1 > 0 && hn >= a.0 + a.1 { (0, a.0 + a.1) } else { a }
}

pub fn next_alive_of(alive: &[bool; MAX_PLAYERS], n: usize, after: usize) -> usize {
    for step in 1..n {
        let q = (after + step) % n;
        if alive[q] {
            return q;
        }
    }
    after
}

/// Liar's Deck as played in Liar's Bar (basic mode).
///
/// Every round: a fresh table rank (Q/K/A) is drawn and the 20-card deck (6 Q, 6 K, 6 A,
/// 2 Jokers) is reshuffled and 5 cards are dealt to each alive player. Jokers always count
/// as the table rank. A turn is either play 1-3 cards face down claiming the table rank, or
/// call the previous play a lie (not allowed on the round's first turn). Players with no
/// cards are skipped. If only one player still holds cards, they must call. The loser of a
/// call pulls their own revolver: 6 chambers, 1 bullet, chamber advances each pull.
///
/// Devil mode: while 3+ players are alive, one of the table-rank cards is the Devil instead of
/// a true card. It can only be played alone. If it is called, the Devil player is safe and the
/// caller plus every other player still holding cards pulls; the Devil player opens next round.
///
/// Devil's Deal (Devil mode; game code: CallSpotOn / ResolveSpotOnRoutine): instead of calling
/// Liar, a player may call Devil's Deal once 4+ cards are in the middle. The last 4 cards are
/// revealed. If all 4 are lies (Jokers and the Devil count as true), every other alive player
/// still holding cards pulls; otherwise the caller pulls. The caller opens the next round.
/// The forced last caller is auto-called Liar and gets no choice.
///
/// Only card counts matter for the rules (true / Devil / other), so hands are stored as counts.
/// Agents must only read their own hand, played_match and played_devil; the rest is public.
/// Public habits of one player, counted over this game plus (in the advisor, and in training
/// for players "met before") earlier games with them. All observable by everyone.
#[derive(Clone, Copy, Default, Debug)]
pub struct Prof {
    /// Chances to call an opening claim, and calls made on one.
    pub open_opps: u16,
    pub open_calls: u16,
    /// Rounds dealt in, and fast dumps (a fresh hand emptied in two plays, 2+3 or 3+2).
    pub rounds: u16,
    pub dumps: u16,
    /// Hand-emptying claims of theirs that were called, and how many were lies.
    pub empty_called: u16,
    pub empty_lies: u16,
    /// Claims of theirs called in 1v1, and how many were lies.
    pub duel_called: u16,
    pub duel_lies: u16,
    /// Claims made, and how many were 3 cards.
    pub claims: u16,
    pub claims3: u16,
}

impl Prof {
    pub fn add(&mut self, o: &Prof) {
        self.open_opps += o.open_opps;
        self.open_calls += o.open_calls;
        self.rounds += o.rounds;
        self.dumps += o.dumps;
        self.empty_called += o.empty_called;
        self.empty_lies += o.empty_lies;
        self.duel_called += o.duel_called;
        self.duel_lies += o.duel_lies;
        self.claims += o.claims;
        self.claims3 += o.claims3;
    }
}

#[derive(Clone)]
pub struct Game {
    pub n: usize,
    pub alive: [bool; MAX_PLAYERS],
    pub shots: [u8; MAX_PLAYERS],
    pub hm: [u8; MAX_PLAYERS],
    pub hn: [u8; MAX_PLAYERS],
    pub hd: [u8; MAX_PLAYERS],
    pub devil_mode: bool,
    /// Ranked lobby (public: the player knows which queue they joined). A network input only.
    pub ranked: bool,
    /// Devil card in play this round (Devil mode and 3+ alive at the deal).
    pub devil_active: bool,
    pub played_devil: [bool; MAX_PLAYERS],
    /// Set by a call that made several players pull (Devil revealed, or a winning Devil's Deal);
    /// `pullers` then lists them and `accused` opens next round.
    pub devil_reveal: bool,
    /// The round ended in a Devil's Deal (won or lost).
    pub deal_called: bool,
    /// Last plays this round, newest first (for Devil's Deal), and how many are stored.
    pub recent: [Play; 4],
    pub n_recent: u8,
    pub accused: usize,
    pub pullers: [bool; MAX_PLAYERS],
    pub claims: [u8; MAX_PLAYERS],
    pub played_match: [u8; MAX_PLAYERS],
    pub claims_total: u8,
    /// Plays made this round so far (the real-call model reads a claim's position in the round).
    pub round_plays: u8,
    pub last: Option<Play>,
    pub current: usize,
    pub starter: usize,
    pub round_over: bool,
    pub loser: usize,
    pub lie: bool,
    pub game_over: bool,
    pub winner: usize,
    /// Public per-player record for the whole game (not reset between rounds):
    /// times their play was called, times that play was a lie, turns where they could call,
    /// and voluntary calls they made.
    pub rec_called: [u8; MAX_PLAYERS],
    pub rec_lies: [u8; MAX_PLAYERS],
    pub rec_opps: [u8; MAX_PLAYERS],
    pub rec_calls: [u8; MAX_PLAYERS],
    /// Voluntary calls that exposed a lie.
    pub rec_calls_right: [u8; MAX_PLAYERS],
    /// Public event history (only kept when `record` is on: the network trainer and advisor).
    pub record: bool,
    pub hist: Vec<Ev>,
    /// Random state for the simulated tells (separate so recording never changes the game).
    pub tell_rng: u64,
    /// Simulated players' own card-check habit (chance to check before a typical claim) and
    /// what an observer knows of it from earlier games: (rate x255, claims seen).
    pub look_habit: [f32; MAX_PLAYERS],
    pub habit_seen: [(u8, u8); MAX_PLAYERS],
    /// Per-player habits (see Prof), and this round's plays per player (count, first size).
    pub prof: [Prof; MAX_PLAYERS],
    pub rd_n: [u8; MAX_PLAYERS],
    pub rd_first: [u8; MAX_PLAYERS],
    /// Seats played by the learning / evaluated network (fitted humans apply the measured bot tax to them).
    pub net_seat: [bool; MAX_PLAYERS],
}

impl Game {
    pub fn new_mode(n: usize, alive: [bool; MAX_PLAYERS], shots: [u8; MAX_PLAYERS], starter: usize, devil_mode: bool, rng: &mut Rng) -> Game {
        let mut g = Game {
            n,
            alive,
            shots,
            hm: [0; MAX_PLAYERS],
            hn: [0; MAX_PLAYERS],
            hd: [0; MAX_PLAYERS],
            devil_mode,
            ranked: false,
            devil_active: false,
            played_devil: [false; MAX_PLAYERS],
            devil_reveal: false,
            deal_called: false,
            recent: [Play { player: 0, n: 0, n_true: 0, devil: false }; 4],
            n_recent: 0,
            accused: 0,
            pullers: [false; MAX_PLAYERS],
            claims: [0; MAX_PLAYERS],
            played_match: [0; MAX_PLAYERS],
            claims_total: 0,
            round_plays: 0,
            last: None,
            current: starter,
            starter,
            round_over: false,
            loser: 0,
            lie: false,
            game_over: false,
            winner: 0,
            rec_called: [0; MAX_PLAYERS],
            rec_lies: [0; MAX_PLAYERS],
            rec_opps: [0; MAX_PLAYERS],
            rec_calls: [0; MAX_PLAYERS],
            rec_calls_right: [0; MAX_PLAYERS],
            record: false,
            hist: Vec::new(),
            tell_rng: rng.next_u64() | 1,
            look_habit: [0.0; MAX_PLAYERS],
            habit_seen: [(0, 0); MAX_PLAYERS],
            prof: [Prof::default(); MAX_PLAYERS],
            rd_n: [0; MAX_PLAYERS],
            rd_first: [0; MAX_PLAYERS],
            net_seat: [false; MAX_PLAYERS],
        };
        g.start_round(starter, rng);
        // Logged lobbies: card-check habits differ hugely between players (most never check
        // while waiting, a few nearly always): rates 3% / 15% / 90% for 70 / 10 / 20% of
        // players match the logged mean 0.23 and spread (sd 0.34). What the others know of a
        // player's habit comes from 0-40 of their earlier claims (half are strangers).
        for q in 0..n {
            let u = rng.f64();
            let h = if u < 0.7 { 0.03 } else if u < 0.8 { 0.15 } else { 0.9 };
            g.look_habit[q] = h as f32;
            let k = if rng.f64() < 0.5 { 0 } else { 1 + rng.below(40) };
            let hits = (0..k).filter(|_| rng.f64() < h).count() as f64;
            let rate = (hits + 2.0 * 0.226) / (k as f64 + 2.0);
            g.habit_seen[q] = ((rate * 255.0).round() as u8, k.min(30) as u8);
        }
        g
    }

    pub fn new_game(n: usize, devil_mode: bool, rng: &mut Rng) -> Game {
        let mut alive = [false; MAX_PLAYERS];
        for a in alive.iter_mut().take(n) {
            *a = true;
        }
        let starter = rng.below(n as u32) as usize;
        Game::new_mode(n, alive, [0; MAX_PLAYERS], starter, devil_mode, rng)
    }

    /// True (table rank or Joker) cards in the deck this round.
    #[inline]
    pub fn truths(&self) -> u8 {
        MATCHING_IN_DECK - self.devil_active as u8
    }

    /// Simulated tells for a claim of `n` cards by `p` (`lie`: any card not as claimed), fitted
    /// to the logged lobbies (nn/fit_tells notes): players check their cards while waiting
    /// mostly before their first play and before bigger claims, somewhat more before lies
    /// (logged 2x, softened here to 1.6x since it is noisy), and lies vary more in think time.
    fn tells(&mut self, p: usize, n: u8, lie: bool) -> (u8, i8, u8, u8) {
        let opening = self.claims_total == 0;
        let before = self.hand(p) + n;
        let habit = self.look_habit[p];
        let st = &mut self.tell_rng;
        let mut next = || {
            *st ^= *st << 13;
            *st ^= *st >> 7;
            *st ^= *st << 17;
            (*st >> 11) as f64 / (1u64 << 53) as f64
        };
        let k = (n.clamp(1, 3) - 1) as usize;
        let mut looks = 0u8;
        if !opening {
            // Situation (first play, claim size) and lie shift the player's own habit, as odds.
            let base: f64 = if before >= 5 { [0.231, 0.304, 0.387][k] } else { [0.156, 0.187, 0.177][k] };
            let extra: f64 = if before >= 5 { 2.3 } else { 0.27 };
            let h = (habit as f64).clamp(0.01, 0.99);
            let odds = h / (1.0 - h) * (base / 0.226) * if lie { 1.3 } else { 0.82 };
            if next() < odds / (1.0 + odds) {
                looks = 1;
                while looks < 6 && next() < extra / (1.0 + extra) {
                    looks += 1;
                }
            }
        }
        let (mu, sd) = if opening { (0.54, 0.67) } else { (-0.107 + 0.07 * k as f64 + if lie { 0.05 } else { 0.0 }, 0.81) };
        let sd = sd * if lie { 1.2 } else { 0.95 };
        let z = (-2.0 * next().max(1e-12).ln()).sqrt() * (std::f64::consts::TAU * next()).cos();
        let pace = ((mu + sd * z).clamp(-3.9, 3.9) * 32.0).round() as i8;
        (looks, pace, self.habit_seen[p].0, self.habit_seen[p].1)
    }

    #[inline]
    pub fn log_ev(&mut self, ev: Ev) {
        if self.record {
            if self.hist.len() >= HIST_KEEP {
                self.hist.remove(0);
            }
            self.hist.push(ev);
        }
    }

    pub fn start_round(&mut self, starter: usize, rng: &mut Rng) {
        for q in 0..self.n {
            self.rd_n[q] = 0;
            if self.alive[q] {
                self.prof[q].rounds = self.prof[q].rounds.saturating_add(1);
            }
        }
        self.devil_active = self.devil_mode && self.n_alive() >= 3;
        let mut rem_total = DECK_SIZE as u32;
        let mut rem_devil = self.devil_active as u32;
        let mut rem_match = MATCHING_IN_DECK as u32 - rem_devil;
        for i in 0..self.n {
            self.hm[i] = 0;
            self.hn[i] = 0;
            self.hd[i] = 0;
            self.claims[i] = 0;
            self.played_match[i] = 0;
            self.played_devil[i] = false;
            if self.alive[i] {
                for _ in 0..HAND_SIZE {
                    let r = rng.below(rem_total);
                    if r < rem_match {
                        self.hm[i] += 1;
                        rem_match -= 1;
                    } else if r < rem_match + rem_devil {
                        self.hd[i] += 1;
                        rem_devil -= 1;
                    } else {
                        self.hn[i] += 1;
                    }
                    rem_total -= 1;
                }
            }
        }
        self.devil_reveal = false;
        self.deal_called = false;
        self.n_recent = 0;
        self.pullers = [false; MAX_PLAYERS];
        self.claims_total = 0;
        self.round_plays = 0;
        self.last = None;
        self.current = starter;
        self.starter = starter;
        self.round_over = false;
        let na = self.n_alive() as u8;
        self.log_ev(Ev { kind: 5, actor: starter as u8, n: na, ..Ev::default() });
    }

    #[inline]
    pub fn hand(&self, p: usize) -> u8 {
        self.hm[p] + self.hn[p] + self.hd[p]
    }

    /// Number of play actions for `p`: the ordinary plays, then the Devil if held.
    #[inline]
    pub fn n_actions(&self, p: usize) -> usize {
        LEGAL[self.hm[p] as usize][self.hn[p] as usize].n + self.hd[p] as usize
    }

    /// Ordinary plays come from LEGAL; DEVIL_PLAY plays the Devil card alone.
    pub fn apply_action(&mut self, a: (u8, u8)) {
        self.round_plays = self.round_plays.saturating_add(1);
        if a == DEVIL_PLAY {
            self.apply_devil();
        } else {
            self.apply_play(a.0, a.1);
        }
    }

    pub fn apply_devil(&mut self) {
        let p = self.current;
        debug_assert!(!self.round_over && !self.is_forced(p) && self.hd[p] == 1);
        if self.last.is_some() {
            self.rec_opps[p] = self.rec_opps[p].saturating_add(1);
        }
        if self.opening_last() {
            self.prof[p].open_opps = self.prof[p].open_opps.saturating_add(1);
        }
        self.hd[p] -= 1;
        self.played_devil[p] = true;
        self.last = Some(Play { player: p, n: 1, n_true: 0, devil: true });
        self.push_recent();
        let (left, total) = (self.hand(p), self.claims_total + 1);
        self.note_claim(p, 1, left);
        if self.record {
            let (looks, pace, habit, habit_n) = self.tells(p, 1, true);
            self.log_ev(Ev { kind: 1, actor: p as u8, n: 1, left, total, looks, pace, habit, habit_n, ..Ev::default() });
        }
        self.claims[p] += 1;
        self.claims_total += 1;
        self.current = self
            .next_active(p)
            .unwrap_or_else(|| next_alive_of(&self.alive, self.n, p));
    }

    #[inline]
    pub fn n_alive(&self) -> usize {
        self.alive[..self.n].iter().filter(|&&a| a).count()
    }

    #[inline]
    pub fn active(&self, q: usize) -> bool {
        self.alive[q] && self.hand(q) > 0
    }

    pub fn next_active(&self, after: usize) -> Option<usize> {
        (1..self.n).map(|s| (after + s) % self.n).find(|&q| self.active(q))
    }

    /// Players other than `a` and `b` that still hold cards.
    #[inline]
    pub fn others_active(&self, a: usize, b: usize) -> usize {
        (0..self.n).filter(|&q| q != a && q != b && self.active(q)).count()
    }

    pub fn is_forced(&self, p: usize) -> bool {
        if self.last.is_none() {
            return false;
        }
        if self.hand(p) == 0 {
            return true;
        }
        !(0..self.n).any(|q| q != p && self.active(q))
    }

    pub fn apply_play(&mut self, t: u8, b: u8) {
        let p = self.current;
        debug_assert!(!self.round_over && !self.is_forced(p));
        debug_assert!((1..=3).contains(&(t + b)) && t <= self.hm[p] && b <= self.hn[p]);
        if self.last.is_some() {
            self.rec_opps[p] = self.rec_opps[p].saturating_add(1);
        }
        if self.opening_last() {
            self.prof[p].open_opps = self.prof[p].open_opps.saturating_add(1);
        }
        self.hm[p] -= t;
        self.hn[p] -= b;
        self.last = Some(Play { player: p, n: t + b, n_true: t, devil: false });
        self.push_recent();
        let (left, total) = (self.hand(p), self.claims_total + t + b);
        self.note_claim(p, t + b, left);
        if self.record {
            let (looks, pace, habit, habit_n) = self.tells(p, t + b, b > 0);
            self.log_ev(Ev { kind: 1, actor: p as u8, n: t + b, left, total, looks, pace, habit, habit_n, ..Ev::default() });
        }
        self.claims[p] += t + b;
        self.played_match[p] += t;
        self.claims_total += t + b;
        self.current = self
            .next_active(p)
            .unwrap_or_else(|| next_alive_of(&self.alive, self.n, p));
    }

    /// The claim on the table is the round's opening claim.
    fn opening_last(&self) -> bool {
        self.last.is_some_and(|l| self.claims_total == l.n)
    }

    /// Habit counts for a claim of `n` cards by `p`, leaving `left`.
    fn note_claim(&mut self, p: usize, n: u8, left: u8) {
        let pr = &mut self.prof[p];
        pr.claims = pr.claims.saturating_add(1);
        if n == 3 {
            pr.claims3 = pr.claims3.saturating_add(1);
        }
        if self.rd_n[p] == 1 && left == 0 && self.rd_first[p] + n == HAND_SIZE {
            pr.dumps = pr.dumps.saturating_add(1);
        }
        if self.rd_n[p] == 0 {
            self.rd_first[p] = n;
        }
        self.rd_n[p] = self.rd_n[p].saturating_add(1);
    }

    fn push_recent(&mut self) {
        self.recent.copy_within(0..3, 1);
        self.recent[0] = self.last.unwrap();
        self.n_recent = (self.n_recent + 1).min(4);
    }

    /// Devil's Deal can be called (Devil mode, 4+ cards in the middle).
    #[inline]
    pub fn deal_legal(&self) -> bool {
        self.devil_mode && self.last.is_some() && self.claims_total >= 4 && !self.is_forced(self.current)
    }

    /// The plays the last 4 cards come from, newest first: (play, cards of it inside the window).
    pub fn deal_window(&self) -> impl Iterator<Item = (Play, u8)> + '_ {
        let mut left = 4u8;
        self.recent[..self.n_recent as usize].iter().filter_map(move |pl| {
            if left == 0 {
                return None;
            }
            let k = pl.n.min(left);
            left -= k;
            Some((*pl, k))
        })
    }

    /// Calls Devil's Deal. Cards inside a play are in random order, so a play only partly inside
    /// the window contributes a random subset of its cards. The Devil card is not a lie.
    pub fn apply_deal(&mut self, rng: &mut Rng) {
        let c = self.current;
        self.rec_opps[c] = self.rec_opps[c].saturating_add(1);
        self.rec_calls[c] = self.rec_calls[c].saturating_add(1);
        // The reveal shows the last 4 cards in throw order, so everyone can tell whose each card
        // is (game code: TakeLast(4) of LastRoundSpotOn). Draw k of each play's n cards without
        // replacement: a play fully inside the window is exposed whole, a partial one shows a subset.
        let window: Vec<(Play, u8)> = self.deal_window().collect();
        let mut all_lies = true;
        let mut reveals: Vec<(Play, u8, u8)> = Vec::with_capacity(4);
        for (pl, k) in window {
            let (mut t, mut n) = ((pl.n_true + pl.devil as u8) as u32, pl.n as u32);
            let mut fakes = 0u8;
            for _ in 0..k {
                if rng.below(n) < t {
                    t -= 1;
                } else {
                    fakes += 1;
                }
                n -= 1;
            }
            if fakes < k {
                all_lies = false;
            }
            reveals.push((pl, k, fakes));
        }
        // Each exposed play counts in the claimer's public record like a called one (a partial
        // reveal only when it shows a fake: real cards in it don't prove the whole play true).
        for &(pl, k, fakes) in &reveals {
            if k == pl.n || fakes > 0 {
                self.rec_called[pl.player] = self.rec_called[pl.player].saturating_add(1);
                if fakes > 0 {
                    self.rec_lies[pl.player] = self.rec_lies[pl.player].saturating_add(1);
                }
            }
        }
        self.pullers = [false; MAX_PLAYERS];
        self.deal_called = true;
        self.lie = all_lies;
        self.loser = c;
        if all_lies {
            self.rec_calls_right[c] = self.rec_calls_right[c].saturating_add(1);
            self.devil_reveal = true;
            self.accused = c;
            for q in 0..self.n {
                self.pullers[q] = q != c && self.alive[q] && self.hand(q) > 0;
            }
        } else {
            self.accused = self.last.unwrap().player;
            self.pullers[c] = true;
        }
        let target = self.last.unwrap().player as u8;
        let total = self.claims_total;
        self.log_ev(Ev { kind: 3, actor: c as u8, target, n: 4, total, flag: all_lies as u8, ..Ev::default() });
        for &(pl, k, fakes) in &reveals {
            self.log_ev(Ev { kind: 6, actor: pl.player as u8, n: k, left: (k == pl.n) as u8, total, flag: fakes, ..Ev::default() });
        }
        self.round_over = true;
    }

    pub fn apply_call(&mut self) {
        let pl = self.last.expect("nothing to call");
        let c = self.current;
        if self.opening_last() {
            self.prof[c].open_opps = self.prof[c].open_opps.saturating_add(1);
            self.prof[c].open_calls = self.prof[c].open_calls.saturating_add(1);
        }
        if !pl.devil {
            let lie = pl.n_true < pl.n;
            let pr = &mut self.prof[pl.player];
            if self.hm[pl.player] + self.hn[pl.player] + self.hd[pl.player] == 0 {
                pr.empty_called = pr.empty_called.saturating_add(1);
                pr.empty_lies = pr.empty_lies.saturating_add(lie as u16);
            }
            if self.alive[..self.n].iter().filter(|&&a| a).count() == 2 {
                pr.duel_called = pr.duel_called.saturating_add(1);
                pr.duel_lies = pr.duel_lies.saturating_add(lie as u16);
            }
        }
        if !self.is_forced(c) {
            self.rec_opps[c] = self.rec_opps[c].saturating_add(1);
            self.rec_calls[c] = self.rec_calls[c].saturating_add(1);
        }
        self.accused = pl.player;
        self.rec_called[pl.player] = self.rec_called[pl.player].saturating_add(1);
        self.pullers = [false; MAX_PLAYERS];
        if pl.devil {
            // The Devil player is safe; the caller and everyone else still holding cards pull.
            self.devil_reveal = true;
            self.lie = false;
            self.loser = c;
            for q in 0..self.n {
                self.pullers[q] = q == c || (q != pl.player && self.alive[q] && self.hand(q) > 0);
            }
        } else {
            self.lie = pl.n_true < pl.n;
            if self.lie {
                self.rec_lies[pl.player] = self.rec_lies[pl.player].saturating_add(1);
                if !self.is_forced(c) {
                    self.rec_calls_right[c] = self.rec_calls_right[c].saturating_add(1);
                }
            }
            self.loser = if self.lie { pl.player } else { c };
            self.pullers[self.loser] = true;
        }
        let flag = if pl.devil { 2 } else { self.lie as u8 };
        let total = self.claims_total;
        self.log_ev(Ev { kind: 2, actor: c as u8, target: pl.player as u8, n: pl.n, total, flag, ..Ev::default() });
        self.round_over = true;
    }

    pub fn next_starter(&self, loser: usize, died: bool, rule: StarterRule, rng: &mut Rng) -> usize {
        match rule {
            // Logged games: the next alive player after the call's loser opens (131 of 132 rounds
            // where the loser survived), whether or not the loser died.
            StarterRule::Loser => {
                let _ = died;
                next_alive_of(&self.alive, self.n, loser)
            }
            StarterRule::Random => {
                let alive: Vec<usize> = (0..self.n).filter(|&q| self.alive[q]).collect();
                alive[rng.below(alive.len() as u32) as usize]
            }
        }
    }

    /// Resolve the revolver pulls and deal the next round. Returns whether the main loser died.
    pub fn finish_round(&mut self, rule: StarterRule, rng: &mut Rng) -> bool {
        let l = self.loser;
        let mut died_main = false;
        for q in 0..self.n {
            if !self.pullers[q] {
                continue;
            }
            let died = rng.below((CHAMBERS - self.shots[q]) as u32) == 0;
            self.shots[q] += 1;
            let left = self.shots[q];
            self.log_ev(Ev { kind: 4, actor: q as u8, left, flag: died as u8, ..Ev::default() });
            if died {
                self.alive[q] = false;
            }
            if q == l {
                died_main = died;
            }
        }
        if self.n_alive() <= 1 {
            self.game_over = true;
            self.winner = (0..self.n).find(|&q| self.alive[q]).unwrap_or(0);
            return died_main;
        }
        let s = if self.devil_reveal { self.accused } else { self.next_starter(l, died_main, rule, rng) };
        self.start_round(s, rng);
        died_main
    }
}
