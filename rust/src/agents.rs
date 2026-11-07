use std::sync::Arc;

use crate::engine::{death_prob, undominate, Game, StarterRule, DECK_SIZE, DEVIL_PLAY, HAND_SIZE, LEGAL};
use crate::policy::{call_key, deal_key, play_key, slack, Policy};
use crate::rng::Rng;

/// Agents may only read public state plus their own hand (hm/hn) and played_match.
pub trait Agent {
    fn decide_call(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool;
    fn decide_play(&mut self, g: &Game, p: usize, rng: &mut Rng) -> (u8, u8);
    /// Call Devil's Deal (only asked when it is legal).
    fn decide_deal(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        deal_heuristic(g, p, 1.0, rng)
    }
    fn miss_stats(&self) -> Option<(u64, u64)> {
        None
    }
    /// For the spot test: whether the forced decision was reached (None for ordinary agents).
    fn as_any_hit(&self) -> Option<bool> {
        None
    }
}

/// Rough chance the last 4 cards are all lies, from card counting only: my own cards in the
/// window are known; each other card is true with the share of unaccounted true cards among
/// the cards others claimed this round (at least what a random card would be).
pub fn deal_p_all_lies(g: &Game, p: usize) -> f64 {
    let unseen = g.truths().saturating_sub(g.hm[p] + g.played_match[p]) as f64;
    let others_claimed = (g.claims_total - g.claims[p]) as f64;
    let q = if others_claimed > 0.0 { (unseen / others_claimed).min(1.0) } else { 0.5 };
    let q = q.max(unseen / (DECK_SIZE - HAND_SIZE) as f64);
    let mut pr = 1.0;
    for (pl, k) in g.deal_window() {
        if pl.player == p {
            let trues = pl.n_true + pl.devil as u8;
            if k + trues > pl.n {
                return 0.0;
            }
            // My play partly in the window: chance none of its true cards landed inside.
            for j in 0..k {
                pr *= (pl.n - trues - j) as f64 / (pl.n - j) as f64;
            }
        } else {
            pr *= (1.0 - q).powi(k as i32);
        }
    }
    pr
}

/// Bots call Devil's Deal when the expected shots dealt to others beat the risk to themselves.
/// `boldness` scales how readily (1 = neutral).
pub fn deal_heuristic(g: &Game, p: usize, boldness: f64, rng: &mut Rng) -> bool {
    let pa = deal_p_all_lies(g, p);
    let others = (g.n_alive() - 1) as f64;
    pa * others * boldness > (1.0 - pa) * 1.5 && rng.f64() < 0.9
}

pub struct RandomAgent {
    pub call_rate: f32,
}

impl Agent for RandomAgent {
    fn decide_call(&mut self, _: &Game, _: usize, rng: &mut Rng) -> bool {
        rng.f32() < self.call_rate
    }

    fn decide_deal(&mut self, _: &Game, _: usize, rng: &mut Rng) -> bool {
        rng.f32() < self.call_rate * 0.15
    }

    fn decide_play(&mut self, g: &Game, p: usize, rng: &mut Rng) -> (u8, u8) {
        let lg = &LEGAL[g.hm[p] as usize][g.hn[p] as usize];
        let a = rng.below(g.n_actions(p) as u32) as usize;
        if a < lg.n { lg.acts[a] } else { DEVIL_PLAY }
    }
}

/// Plays true cards when it has them, bluffs one card when it doesn't, calls on hard evidence.
pub struct HonestAgent;

impl Agent for HonestAgent {
    fn decide_deal(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        deal_heuristic(g, p, 0.5, rng)
    }

    fn decide_call(&mut self, g: &Game, p: usize, _: &mut Rng) -> bool {
        let s = slack(g, p);
        s < 0 || (g.last.unwrap().n >= 2 && s <= 1)
    }

    fn decide_play(&mut self, g: &Game, p: usize, rng: &mut Rng) -> (u8, u8) {
        let m = g.hm[p];
        if m > 0 {
            (m.min(1 + rng.below(2) as u8), 0)
        } else if g.hn[p] > 0 {
            (0, 1)
        } else {
            DEVIL_PLAY
        }
    }
}

fn comb(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let mut r = 1u64;
    for i in 0..k {
        r = r * (n - i) / (i + 1);
    }
    r
}

pub fn hypergeom_at_least(k: u64, population: u64, successes: u64, draws: u64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if successes < k || draws < k {
        return 0.0;
    }
    let total = comb(population, draws) as f64;
    let mut acc = 0u64;
    for x in k..=draws.min(successes) {
        acc += comb(successes, x) * comb(population - successes, draws - x);
    }
    acc as f64 / total
}

/// Rough probability the claimer's cumulative claims this round are all true, treating their
/// dealt hand as a random 5 cards from everything not in my dealt hand.
pub fn p_claim_true(g: &Game, p: usize) -> f64 {
    let prev = g.last.unwrap().player;
    let need = g.claims[prev] as u64;
    let remaining = (g.truths() as i64 - g.hm[p] as i64 - g.played_match[p] as i64).max(0) as u64;
    hypergeom_at_least(need, (DECK_SIZE - HAND_SIZE) as u64, remaining, HAND_SIZE as u64)
}

/// Hypergeometric lie estimate with a threshold shifted by both players' revolver odds.
pub struct ProbAgent {
    pub base_threshold: f64,
    pub risk_weight: f64,
    pub bluff_rate: f32,
}

impl Default for ProbAgent {
    fn default() -> Self {
        ProbAgent { base_threshold: 0.5, risk_weight: 0.4, bluff_rate: 0.15 }
    }
}

impl Agent for ProbAgent {
    fn decide_deal(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        deal_heuristic(g, p, 1.6 - self.base_threshold, rng)
    }

    fn decide_call(&mut self, g: &Game, p: usize, _: &mut Rng) -> bool {
        let lp = g.last.unwrap();
        if slack(g, p) < 0 && lp.n >= 2 {
            return true;
        }
        let lie_p = 1.0 - p_claim_true(g, p);
        let d_me = death_prob(g.shots[p]);
        let d_them = death_prob(g.shots[lp.player]);
        let thr = self.base_threshold + self.risk_weight * (d_me - d_them);
        lie_p > thr.clamp(0.05, 0.95)
    }

    fn decide_play(&mut self, g: &Game, p: usize, rng: &mut Rng) -> (u8, u8) {
        let (m, nm) = (g.hm[p], g.hn[p]);
        if g.hd[p] > 0 && (m + nm == 0 || rng.f32() < self.bluff_rate) {
            return DEVIL_PLAY;
        }
        if m > 0 {
            let t = m.min(if m + nm <= 3 { 3 } else { 2 });
            if t < 3 && nm > 0 && rng.f32() < self.bluff_rate {
                return (t, 1);
            }
            return (t, 0);
        }
        if nm >= 2 && rng.f32() < self.bluff_rate {
            return (0, 2);
        }
        (0, 1)
    }
}

/// Calls the way real lobby players do (about 28% of chances in logged games): rarely on solid
/// claims, more on unlikely ones and on big claims. Plays true cards, bluffs now and then.
pub struct HumanAgent {
    /// Scales every call probability (1.0 is about the logged human rate).
    pub call_scale: f64,
    /// Chance to bluff while still holding real cards.
    pub bluff_rate: f32,
    /// Chance a truthful play throws as many real cards as allowed (up to 3) instead of 1-2.
    pub big: f32,
    /// Chance a bluff is 2 cards, and 3 cards.
    pub bluff2: f32,
    pub bluff3: f32,
    /// Card saving: bluff early while the hand is big, keep real cards to finish with, and
    /// stay honest near the end (logged lobbies: claims leaving 3-4 cards were lies 47% of the
    /// time, leaving 0-2 only 25%).
    pub save: f32,
    /// How much more readily they call opening claims than the logged average (1.0). Some
    /// players (WeirderLime-style) call openers constantly, especially in 1v1.
    pub open_call: f64,
    /// Ranked-lobby player (Devil mode only): uses the fit to ranked Devil logs, which bluff far
    /// less (called plays lies 27% on 2026-10-31). Added alongside the old archetypes, not instead.
    pub ranked: bool,
    /// This player's own calling level, as the real-call model sees it: their share of chances
    /// taken. Real opponents (90 with 15+ chances, 2026-11-01): mean 0.20, sd 0.11, top decile 0.37+.
    pub rate_trait: f64,
}

/// Real-call model (nn/call_model.json, 2026-11-01): logistic fit to 4,894 real Devil claims that
/// had a free next actor; label = that player called. Held-out AUC 0.73, calibrated. Inputs the
/// simulator cannot know (timing, slots, the claimer's history length) sit at their means.
const RC_W: [f64; 24] = [-1.5612, 0.0753, -0.2081, 0.2259, 0.1796, 0.1397, -0.1352, -0.3345, 0.0578, 0.0513, -0.0117, -0.0776, -0.0558, -0.1571, -0.0398, 0.0303, -0.0522, -0.0539, 0.0331, 0.0691, 0.0697, 0.0352, 0.1745, -0.2157];
const RC_MU: [f64; 24] = [0.0, 1.3224, 2.9240, 3.5995, 0.7583, 3.4593, 3.2346, 3.1353, 1.8152, 0.9961, -0.1438, 0.3469, 3.9865, 0.3057, 0.5971, 0.8480, 0.2043, 0.1105, 0.0544, 0.2117, 0.0715, 0.1965, 0.1946, 3.2844];
const RC_SD: [f64; 24] = [1.0, 0.6013, 1.1270, 2.2022, 0.9921, 3.0892, 0.7483, 0.7967, 0.5253, 0.0622, 0.5594, 0.4618, 1.0721, 0.4607, 0.4905, 1.2503, 0.4032, 0.3136, 0.2267, 0.4085, 0.2577, 0.0511, 0.0726, 2.2384];

/// Chance a real ranked player with calling level `rate_trait` calls the last claim, from the
/// public situation only.
pub fn real_call_prob(g: &Game, p: usize, rate_trait: f64) -> f64 {
    let lp = g.last.unwrap();
    let c = lp.player;
    let left = g.hand(c) as f64;
    let holders = (0..g.n).filter(|&q| g.active(q)).count() as f64;
    let idx = (g.round_plays.max(1) as f64).min(8.0);
    let earlier = g.claims[c].saturating_sub(lp.n) as f64;
    let total_before = (g.claims_total.saturating_sub(lp.n) as f64).min(12.0);
    let x = [
        1.0, lp.n as f64, left, idx, earlier, total_before, g.n_alive() as f64, holders,
        RC_MU[8], 1.0, RC_MU[10], RC_MU[11],            // timing: unknown here, neutral
        g.hand(p) as f64, 0.0,                           // next's cards; next is not the logging player
        RC_MU[14], RC_MU[15], RC_MU[16], RC_MU[17],      // slots: unknown here, neutral
        (left == 0.0) as u8 as f64, (idx == 1.0) as u8 as f64, (lp.n >= 3) as u8 as f64,
        rate_trait, rate_trait, RC_MU[23],
    ];
    let mut z = 0.0;
    for j in 0..24 {
        z += RC_W[j] * (x[j] - RC_MU[j]) / RC_SD[j];
    }
    let pr = 1.0 / (1.0 + (-z).exp());
    if std::env::var("LIARS_RC_DEBUG").is_ok() && left == 0.0 {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 12 {
            eprintln!("rc: n={} left={} idx={} earlier={} total_before={} alive={} holders={} next_cards={} trait={:.2} -> p={:.2}",
                lp.n, left, idx, earlier, total_before, g.n_alive(), holders, g.hand(p), rate_trait, pr);
        }
    }
    pr
}

/// Corrections fitted to the logged lobbies (nn/fit_humans.py against nn/real_stats.py), as
/// multipliers on the hand-set behaviour: [all calls, calls on a claimer left with 1-2 cards,
/// opening calls, late bluffs, early bluff rate, extra calls on big claims, 1v1 calls, calls once
/// 7+ cards are claimed in the round, bluffing the whole small hand out].
/// LIARS_HUMAN_FIT="a,b,..." overrides them (for the fit itself).
pub const HUMAN_FIT_DEFAULT: [f64; 13] = [0.196, 1.084, 1.537, 2.117, 1.0, 0.554, 1.462, 0.588, 0.35, 0.0, 1.0, 2.0, 1.0];
/// Devil mode has its own fit (ranked Devil lobbies bluff far less: called plays were lies 27% of
/// the time on 2026-10-31 against 47% in the old simulation). LIARS_HUMAN_FIT_DEVIL overrides it.
pub const HUMAN_FIT_DEVIL: [f64; 13] = [0.196, 1.084, 1.537, 2.117, 1.0, 0.554, 1.462, 0.588, 0.35, 0.0, 1.0, 2.0, 1.0];

fn load_fit(default: [f64; 13], var: &str) -> [f64; 13] {
    let mut f = default;
    if let Ok(v) = std::env::var(var) {
        for (i, x) in v.split(',').enumerate().take(13) {
            if let Ok(x) = x.trim().parse() {
                f[i] = x;
            }
        }
    }
    f
}

/// Fit to ranked Devil lobbies (nn/fit_humans.py --ranked). LIARS_HUMAN_FIT_RANKED overrides it;
/// LIARS_ALL_RANKED=1 makes every human-like player ranked (for the fit itself).
/// 2026-11-01 refit (nn/fit_ranked6.log) over all Devil logs with the real-call model as the
/// callers and the junk-first play order; slots 3/4/8 scale late/early/whole-hand bluffing,
/// 10 the call odds, 11 multi-card bluff share, 12 multi-card real-play share (9 is unused now).
/// Held-out: calls within a few points everywhere, lies still ~10 points above the logs overall.
pub const HUMAN_FIT_RANKED: [f64; 13] = [0.043, 2.946, 1.323, 0.259, 1.822, 0.304, 1.462, 1.245, 0.292, 0.500, 0.607, 1.300, 0.932];

pub fn ranked_fit() -> &'static [f64; 13] {
    static R: std::sync::OnceLock<[f64; 13]> = std::sync::OnceLock::new();
    R.get_or_init(|| load_fit(HUMAN_FIT_RANKED, "LIARS_HUMAN_FIT_RANKED"))
}

fn all_ranked() -> bool {
    static A: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *A.get_or_init(|| std::env::var("LIARS_ALL_RANKED").map(|v| v == "1").unwrap_or(false))
}

impl HumanAgent {
    fn fit(&self, g: &Game) -> &'static [f64; 13] {
        if g.devil_mode && (self.ranked || all_ranked()) { ranked_fit() } else { human_fit(g.devil_mode) }
    }
}

pub fn human_fit(devil: bool) -> &'static [f64; 13] {
    static BASIC: std::sync::OnceLock<[f64; 13]> = std::sync::OnceLock::new();
    static DEVIL: std::sync::OnceLock<[f64; 13]> = std::sync::OnceLock::new();
    if devil {
        DEVIL.get_or_init(|| load_fit(HUMAN_FIT_DEVIL, "LIARS_HUMAN_FIT_DEVIL"))
    } else {
        BASIC.get_or_init(|| load_fit(HUMAN_FIT_DEFAULT, "LIARS_HUMAN_FIT"))
    }
}

/// Logistic call model fitted to 3,254 real call-or-pass decisions by logged players
/// (nn/callfit_export.py; public inputs only, no timing). Inputs: claim size, claimer left 0,
/// claimer left 1-2, decider's cards, 1v1, players holding cards, cards claimed this round,
/// opening claim, claimer's pulls, decider's pulls, times the claimer was called, claimer's
/// caught-lying rate, decider's call rate, Devil mode, ranked.
pub const CALL_W: [f64; 15] = [0.13400, 0.19342, 0.39040, 0.05970, 0.48200, -0.23775, 0.11761, 0.00902, -0.06279, -0.17249, 0.12253, 0.39944, 0.02959, 0.03034, 0.08170];
pub const CALL_B: f64 = -1.72704;

pub fn learned_call_logit(g: &Game, p: usize, ranked: bool) -> f64 {
    let lp = g.last.unwrap();
    let c = lp.player;
    let left = g.hand(c);
    let holders = (0..g.n).filter(|&q| g.alive[q] && g.hand(q) > 0).count();
    let called = g.rec_called[c].min(8) as f64;
    let x = [
        lp.n as f64,
        (left == 0) as u8 as f64,
        (1..=2).contains(&left) as u8 as f64,
        g.hand(p) as f64,
        (g.n_alive() == 2) as u8 as f64,
        holders as f64,
        g.claims_total as f64,
        (g.claims_total == lp.n) as u8 as f64,
        g.shots[c] as f64,
        g.shots[p] as f64,
        called,
        (g.rec_lies[c].min(8) as f64 + 0.5) / (called + 2.0),
        (g.rec_calls[p] as f64 + 0.5) / (g.rec_opps[p] as f64 + 2.0),
        g.devil_mode as u8 as f64,
        ranked as u8 as f64,
    ];
    CALL_B + CALL_W.iter().zip(x.iter()).map(|(w, v)| w * v).sum::<f64>()
}

impl Default for HumanAgent {
    /// The "average" archetype; its claim sizes match logged lobbies (about 45% 1-card,
    /// 36% 2-card, 19% 3-card claims).
    fn default() -> Self {
        HumanAgent { call_scale: 1.0, bluff_rate: 0.22, big: 0.6, bluff2: 0.4, bluff3: 0.1, save: 0.6, open_call: 1.0, ranked: false, rate_trait: 0.20 }
    }
}

impl HumanAgent {
    /// A random human-like player: every trait drawn independently over a wide range, so the
    /// logged archetypes (honest big-claimer, average, bluffer) are regions of the space rather
    /// than the only options. Averages match logged lobbies (bluff rate about 0.21).
    pub fn random_traits(call_scale: f64, rng: &mut Rng) -> HumanAgent {
        let u = rng.f32();
        let v = rng.f32();
        // Calling level drawn to match the real spread (mean 0.20, sd 0.11; three uniforms
        // approximate a normal), clipped to the range seen in the logs.
        let z = (rng.f64() + rng.f64() + rng.f64() - 1.5) / 0.5;
        HumanAgent {
            call_scale,
            bluff_rate: 0.02 + 0.58 * u * u,
            big: 0.2 + 0.7 * rng.f32(),
            bluff2: 0.1 + 0.5 * rng.f32(),
            bluff3: 0.3 * v * v,
            save: rng.f32(),
            open_call: 1.0,
            ranked: false,
            rate_trait: (0.20 + 0.11 * z).clamp(0.03, 0.55),
        }
    }

    /// A heavy caller: calls far more than average, openers included (WeirderLime-style).
    /// Used by the Python training bridge and simstats.
    pub fn heavy_caller(rng: &mut Rng) -> HumanAgent {
        HumanAgent {
            call_scale: 2.0 + 1.5 * rng.f64(),
            open_call: 2.0 + 4.0 * rng.f64(),
            rate_trait: 0.35 + 0.15 * rng.f64(), // the real top decile of callers
            ..HumanAgent::random_traits(1.0, rng)
        }
    }
}

/// How readily human-like players call Devil's Deal, fitted so simulated deals happen as often
/// as in the logs (LIARS_DEAL_SCALE overrides it, for the fit). Logs: deals that show (won, or
/// lost on a fake last play) end 1.2% of Devil rounds (5 of 417); simstats gives 1.0% at 4 and
/// 1.4% at 5, and 0.05% at the old 1.
pub fn deal_scale() -> f64 {
    static S: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *S.get_or_init(|| std::env::var("LIARS_DEAL_SCALE").ok().and_then(|v| v.parse().ok()).unwrap_or(4.5))
}

impl Agent for HumanAgent {
    fn decide_deal(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        deal_heuristic(g, p, self.call_scale * deal_scale(), rng)
    }

    fn decide_call(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        let lp = g.last.unwrap();
        // (Ranked players skip the "impossible total" shortcut below: the real-call model was
        // fitted on real players' responses to those claims too, and real players miss them.)
        if slack(g, p) < 0 && lp.n >= 2 && !(g.devil_mode && (self.ranked || all_ranked())) {
            return true;
        }
        // Ranked players call exactly as the real-call model says real players do (2026-11-01):
        // public situation plus their own calling level. Nothing private is added on top: the old
        // doubt / no-real-cards / call_scale terms stacked on a model already fitted to the real
        // call rate and made the simulated population call about twice as often as the logs.
        // fit[10] remains as a global odds scale (1.0 = as fitted).
        if g.devil_mode && (self.ranked || all_ranked()) {
            let fit = self.fit(g);
            let pr = real_call_prob(g, p, self.rate_trait);
            // Out of real cards with a small hand, a real player would rather call than be left
            // faking (the logs' late plays are far more honest than forced faking would make them).
            let private = if g.hm[p] == 0 && g.hand(p) <= 3 { 1.6 } else { 1.0 };
            let odds = pr / (1.0 - pr).max(1e-9) * fit[10] * private;
            return rng.f64() < (odds / (1.0 + odds)).clamp(0.0, 0.95);
        }
        let doubt = 1.0 - p_claim_true(g, p);
        // Fitted to logged lobbies: opening 1-card claims are called ~11% of the time, later
        // claims ~25%, bigger and less plausible claims more often.
        let opening = g.claims_total == lp.n;
        let fit = self.fit(g);
        let base = if opening && lp.n == 1 { 0.03 * self.open_call } else if opening { 0.10 * self.open_call.sqrt() } else { 0.10 };
        let mut pc = (base + 0.45 * doubt + 0.06 * fit[5] * (lp.n as f64 - 1.0)) * self.call_scale * fit[0];
        // Logged lobbies call more as the table shrinks: 20% of chances at 4 players, 23% at 3,
        // 32% in 1v1.
        pc *= match g.n_alive() {
            2 => 1.5 * fit[6],
            3 => 1.1,
            _ => 1.0,
        };
        // Out of real cards late in the round, savers would rather call than bluff.
        if g.hm[p] == 0 && g.hand(p) <= 3 {
            pc = pc.max(0.4) * (1.5 + 2.0 * self.save as f64);
        }
        // Fitted situation corrections, applied last so nothing above overrides them.
        if opening {
            pc *= fit[2];
        }
        if (1..=2).contains(&g.hand(lp.player)) {
            pc *= fit[1];
        }
        if g.claims_total >= 7 {
            pc *= fit[7];
        }
        // Ranked fit only (0 elsewhere): with no real cards, call rather than be forced to bluff.
        if g.hm[p] == 0 && fit[9] > 0.0 {
            pc = pc.max(fit[9]);
        }
        rng.f64() < pc.clamp(0.0, 0.95)
    }

    fn decide_play(&mut self, g: &Game, p: usize, rng: &mut Rng) -> (u8, u8) {
        let (m, nm) = (g.hm[p], g.hn[p]);
        if g.hd[p] > 0 && (m + nm == 0 || rng.f32() < 0.15) {
            return DEVIL_PLAY;
        }
        // Ranked players (2026-11-01, real logs): bluff while the hand is big and finish honest.
        // Called opening claims were lies 54% of the time, called hand-emptying throws 33%, called
        // plays leaving 1-2 cards 40%. The old play order did the reverse (real cards thrown early,
        // fakes left for the end: 72% lies on called dumps), which taught the networks to call
        // late plays. fit[4] scales the early bluff rate, fit[3] the late one.
        if g.devil_mode && (self.ranked || all_ranked()) {
            let fit = self.fit(g);
            let hand = m + nm;
            // Junk goes out in twos and threes (logged claims: 67% one card, 23% two, 9% three);
            // single-card bluffing would make three of every five plays a lie. fit[11] scales it.
            let bs = fit[11] as f32;
            let bluff_size = |nm: u8, rng: &mut Rng| -> u8 {
                let r = rng.f32();
                if nm >= 3 && r < 0.08 * bs { 3 } else if nm >= 2 && r < 0.38 * bs { 2 } else { 1 }
            };
            // Real cards also go out in twos and threes (logged 2- and 3-card claims are true
            // more often than not: lies 43% and 45% when called), but while junk remains at least
            // one real card is kept for the finish. fit[12] scales the multi-card share.
            let rs = fit[12] as f32;
            let real_size = |m: u8, nm: u8, rng: &mut Rng| -> u8 {
                let r = rng.f32();
                let cap = if nm > 0 { m.saturating_sub(1).max(1) } else { m };
                let k = if m >= 3 && r < 0.25 * rs { 3 } else if m >= 2 && r < 0.5 * rs { 2 } else { 1 };
                k.min(cap).min(3)
            };
            if hand >= 4 {
                // Early: bluff often when holding junk (logged claim sizes 67/23/9%: mostly
                // singles, some pairs), otherwise one or two real cards.
                // The opener bluffs often (logged openings: 54% lies when called); the second play
                // of a hand far less (plays leaving 3-4 cards: 36%).
                let base = if g.claims_total == 0 { 0.55 } else { 0.15 };
                let early = (base * fit[4] as f32 * (0.6 + 0.8 * self.bluff_rate)).clamp(0.05, 0.9);
                // Dumpers sometimes open with three cards, real ones first (logged 3-card claims
                // are 9% of all claims and lies 45% when called).
                if hand >= 5 && rng.f32() < 0.12 * (1.0 - self.save) {
                    let t = m.min(3);
                    if nm >= 3 - t { return (t, 3 - t); }
                }
                // Savers keep a lone real card for the finish and play the junk first; others
                // spend it when it comes up.
                let keep_last = m == 1 && rng.f32() < 0.5 + 0.5 * self.save;
                if nm > 0 && (m == 0 || keep_last || rng.f32() < early) {
                    return (0, bluff_size(nm, rng));
                }
                return (real_size(m, nm, rng), 0);
            }
            // Late (3 or fewer cards). Dumpers (low save) throw the whole 3-card hand, fakes
            // included (logged 3-card claims: 9% of claims, lies 45% when called). All real: throw
            // them all. Mixed: savers play the junk now and keep a real card for the finish;
            // others throw real cards, all at once when big.
            if hand == 3 && nm > 0 && rng.f32() < 0.03 * (1.0 - self.save) { return (m, nm); }
            if m > 0 {
                if nm == 0 { return (m.min(3), 0); }
                // Savers play the junk now so the hand ends on real cards (logged hand-emptying
                // throws are true two times in three).
                if rng.f32() < 0.6 * self.save * fit[3] as f32 { return (0, bluff_size(nm, rng)); }
                return (if rng.f32() < self.big { m.min(3) } else { real_size(m, nm, rng) }, 0);
            }
            // No real card left: fakes, sometimes the whole small hand.
            if nm <= 3 && rng.f32() < 0.1 * fit[8] as f32 { return (0, nm); }
            return (0, bluff_size(nm, rng));
        }
        // Emptying the hand with only real cards (forcing the next player to call) is the
        // logged players' most reliable move: 13 of 15 called hand-dumps were true.
        if nm == 0 && m <= 3 && rng.f32() < 0.9 {
            return (m, 0);
        }
        let hand = m + nm;
        let fit = self.fit(g);
        // Logged lobbies: called hand-dumps that empty the hand are lies about half the time, so
        // some players throw their whole small hand, fakes included.
        if nm > 0 && hand <= 3 && rng.f64() < 0.1 * fit[8] {
            return (m, nm);
        }
        // Savers bluff more while the hand is big and less once it is small.
        let early = (self.bluff_rate * fit[4] as f32).min(0.9);
        let rate = if hand >= 5 {
            // fit[4] also scales card-saving bluffs (1.0 in the Basic and old Devil fits).
            early + self.save * 0.35 * (1.0 - early) * fit[4] as f32
        } else {
            // Logged players rarely bluff by choice near the end of their hand.
            (self.bluff_rate * 0.25 * fit[3] as f32 * (1.0 - 0.8 * self.save)).min(0.9)
        };
        let bluff = m == 0 || (nm > 0 && rng.f32() < rate);
        if !bluff {
            // Savers hold real cards back for the finish instead of throwing them all at once.
            let early = hand >= 5;
            let big = self.big * (1.0 - 0.4 * self.save * early as u8 as f32);
            let k = if rng.f32() < big {
                m.min(3)
            } else if early && rng.f32() < self.save {
                1
            } else {
                m.min(1 + rng.below(2) as u8)
            };
            return (k, 0);
        }
        let r = rng.f32();
        let mut b = if r < self.bluff3 { 3 } else if r < self.bluff3 + self.bluff2 { 2 } else { 1 };
        // With only fake cards left late, savers bluff one card at a time rather than dump them.
        if m == 0 && hand <= 3 && rng.f32() < self.save {
            b = 1;
        }
        b = b.min(nm);
        // Sometimes a real card goes along with the bluff ("K K Joker").
        let t = if m > 0 && b < 3 && hand >= 5 && rng.f32() < 0.2 { 1 } else { 0 };
        (t, b)
    }
}

/// A fast dumper: often empties a fresh hand in two plays, 2 then 3 or 3 then 2 cards, fakes in
/// the first throw and the real cards kept for the last one. Logged lobbies: some players do this
/// in 40-50% of their rounds (about 6% overall), and the final throw, when called, was true about
/// 70% of the time (lies 20% after 2-then-3, 44% after 3-then-2). Calls like the human-like
/// player it wraps.
pub struct DumperAgent {
    pub inner: HumanAgent,
    /// Chance a fresh 5-card hand with 2+ real cards is dumped this way (a fifth as often with less).
    pub dump: f32,
    /// Chance the first throw is 3 cards when the hand doesn't decide it.
    pub three_first: f32,
    /// Munk-style: real cards go first, so the final throw is usually fakes (a lie); and the
    /// hand is dumped whatever it holds.
    pub liar: bool,
    /// Dumping this round (decided when the hand is fresh).
    plan: bool,
}

impl DumperAgent {
    pub fn new(inner: HumanAgent, dump: f32, three_first: f32) -> Self {
        DumperAgent { inner, dump, three_first, liar: false, plan: false }
    }

    pub fn random(rng: &mut Rng) -> Self {
        let cs = 0.4 + 2.1 * rng.f64() * rng.f64();
        let mut d = DumperAgent::new(HumanAgent::random_traits(cs, rng), 0.4 + 0.5 * rng.f32(), rng.f32());
        d.liar = rng.f32() < 0.5;
        d
    }
}

impl Agent for DumperAgent {
    fn decide_deal(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        self.inner.decide_deal(g, p, rng)
    }

    fn decide_call(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        self.inner.decide_call(g, p, rng)
    }

    fn decide_play(&mut self, g: &Game, p: usize, rng: &mut Rng) -> (u8, u8) {
        let (m, nm) = (g.hm[p], g.hn[p]);
        let hand = m + nm;
        if g.hd[p] == 0 {
            if hand == HAND_SIZE {
                let w = if m >= 2 || self.liar { 1.0 } else { 0.2 };
                self.plan = rng.f32() < self.dump * w;
                if self.plan {
                    // Keep enough real cards for the last throw: 3 real -> 2 first, 2 real -> 3
                    // first; now and then the player's habit wins instead.
                    let k = if self.liar {
                        if rng.f32() < self.three_first { 3 } else { 2 }
                    } else if rng.f32() < 0.8 && m >= 2 {
                        if m >= 3 { 2 } else { 3 }
                    } else if rng.f32() < self.three_first {
                        3
                    } else {
                        2
                    };
                    if self.liar {
                        let t = m.min(k);
                        return (t, k - t);
                    }
                    let f = nm.min(k);
                    return (k - f, f);
                }
            } else if self.plan && hand <= 3 {
                self.plan = false;
                return (m, nm);
            }
        }
        self.plan = false;
        self.inner.decide_play(g, p, rng)
    }
}

/// A reader (Jaime-style, the #2 ranked player in a logged game): plays like the human-like
/// player it wraps, but once someone has been caught bluffing it keeps calling that player's
/// claims, small ones and openers included (2 of 3 such calls were right, one killed).
pub struct ReaderAgent {
    pub inner: HumanAgent,
    /// Extra call chance per time the claimer was caught lying (capped).
    pub grudge: f64,
}

impl ReaderAgent {
    pub fn random(rng: &mut Rng) -> Self {
        let cs = 0.4 + 2.1 * rng.f64() * rng.f64();
        ReaderAgent { inner: HumanAgent::random_traits(cs, rng), grudge: 0.15 + 0.25 * rng.f64() }
    }
}

impl Agent for ReaderAgent {
    fn decide_deal(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        self.inner.decide_deal(g, p, rng)
    }

    fn decide_call(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        if self.inner.decide_call(g, p, rng) {
            return true;
        }
        let lp = g.last.unwrap();
        let (lies, called) = (g.rec_lies[lp.player] as f64, g.rec_called[lp.player] as f64);
        if lies == 0.0 {
            return false;
        }
        // Grudge grows with how often they were caught, fades if their called plays were true.
        let rate = (lies + 1.0) / (called + 2.0);
        let extra = (self.grudge * lies.min(3.0) * (rate / 0.5)).min(0.6);
        rng.f64() < extra * (1.0 - p_claim_true(g, p)).max(0.3)
    }

    fn decide_play(&mut self, g: &Game, p: usize, rng: &mut Rng) -> (u8, u8) {
        self.inner.decide_play(g, p, rng)
    }
}

pub struct CfrAgent {
    pub policy: Arc<Policy>,
    pub fallback: ProbAgent,
    pub lookups: u64,
    pub misses: u64,
}

impl CfrAgent {
    pub fn new(policy: Arc<Policy>) -> Self {
        CfrAgent { policy, fallback: ProbAgent::default(), lookups: 0, misses: 0 }
    }
}

impl Agent for CfrAgent {
    fn decide_deal(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        match self.policy.call_probs(deal_key(g, p)) {
            Some(pr) => rng.sample(pr, 2) == 1,
            None => deal_heuristic(g, p, 1.0, rng),
        }
    }

    fn decide_call(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        self.lookups += 1;
        match self.policy.call_probs(call_key(g, p, &self.policy.feat())) {
            Some(pr) => rng.sample(pr, 2) == 1,
            None => {
                self.misses += 1;
                self.fallback.decide_call(g, p, rng)
            }
        }
    }

    fn decide_play(&mut self, g: &Game, p: usize, rng: &mut Rng) -> (u8, u8) {
        self.lookups += 1;
        let lg = &LEGAL[g.hm[p] as usize][g.hn[p] as usize];
        match self.policy.play_probs(play_key(g, p, &self.policy.feat())) {
            Some(pr) => {
                let a = rng.sample(pr, g.n_actions(p));
                if a < lg.n { undominate(g.hn[p], lg.acts[a]) } else { DEVIL_PLAY }
            }
            None => {
                self.misses += 1;
                self.fallback.decide_play(g, p, rng)
            }
        }
    }

    fn miss_stats(&self) -> Option<(u64, u64)> {
        Some((self.misses, self.lookups))
    }
}


/// A ranked human fitted to the logs (research/fit_population.py -> fit_data.rs, 2026-11-06):
/// plays by the empirical table P(action | real, junk, Devil card, opener, 1v1) over 6,601 fully
/// observed human plays, with a personal bluffing level; calls by a logistic model of the public
/// situation plus its own hand (humans holding no real card call 41% of chances, holding 2+ about
/// 10%) and a personal calling level. Held-out AUC 0.76 on 7,934 real call decisions.
pub struct FitHuman {
    /// Log-odds offset on bluffing (population sd FIT_BLUFF_SD).
    pub bluff_z: f64,
    /// Share of chances this player calls (population mean 0.22, sd 0.055).
    pub rate_trait: f64,
    /// State-blind share: this player's call probability is (1-blind) * model + blind * flat trait rate. Real humans'
    /// calls are only partly predictable from the public state (AUC 0.77); a fully state-driven population has
    /// blind spots a trained policy learns to bluff into that real tables do not have.
    pub blind: f64,
    /// Pattern reader (learned over earlier games against the bot): log-odds split added on network seats' claims,
    /// +read on lies and -read on truths. Live: callers with 2+ prior games against the bot keep the same call rate but
    /// their precision rises from 0.61 to 0.72, which +/-0.25 reproduces at the bot's 0.67 bluff rate.
    pub read: f64,
}

/// Share of fitted humans that are pattern readers (LIARS_FIT_READ_SHARE, default 0) and their strength (LIARS_FIT_READ, 0.25).
pub fn fit_read_params() -> (f64, f64) {
    static P: std::sync::OnceLock<(f64, f64)> = std::sync::OnceLock::new();
    *P.get_or_init(|| (
        std::env::var("LIARS_FIT_READ_SHARE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0),
        std::env::var("LIARS_FIT_READ").ok().and_then(|v| v.parse().ok()).unwrap_or(0.25),
    ))
}

/// Extra log-odds real humans put on calling the bot's claims over a human making the same claims
/// (research/fit_population.py, 2026-11-06 05:50 refit: 29.2% called vs 26.1% predicted on 4,392 live claims). LIARS_FIT_TAX overrides.
pub fn fit_bot_tax() -> f64 {
    static T: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("LIARS_FIT_TAX").ok().and_then(|v| v.parse().ok()).unwrap_or(0.154))
}

fn approx_normal(rng: &mut Rng) -> f64 {
    (0..12).map(|_| rng.f64()).sum::<f64>() - 6.0
}

impl FitHuman {
    pub fn random(rng: &mut Rng) -> FitHuman {
        FitHuman {
            bluff_z: approx_normal(rng) * crate::fit_data::FIT_BLUFF_SD,
            rate_trait: (crate::fit_data::FIT_TRAIT_MEAN + approx_normal(rng) * crate::fit_data::FIT_TRAIT_SD).clamp(0.03, 0.6),
            blind: if rng.f64() < 0.3 { 0.5 } else { 0.0 },
            read: { let (share, r) = fit_read_params(); if rng.f64() < share { r } else { 0.0 } },
        }
    }

    /// For make_agent (no rng at hand): a fresh random player each time.
    pub fn random_global() -> FitHuman {
        static SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0x5eed_f17);
        let s = SEED.fetch_add(0x9E37_79B9_7F4A_7C15, std::sync::atomic::Ordering::Relaxed);
        FitHuman::random(&mut Rng::new(s))
    }

    /// Chance a fitted human in seat `p` calls the last claim.
    pub fn call_prob(g: &Game, p: usize, rate_trait: f64) -> f64 {
        use crate::fit_data::*;
        let lp = g.last.unwrap();
        let c = lp.player;
        let n = lp.n as f64;
        let left = g.hand(c) as f64;
        let plays_before = (g.round_plays.saturating_sub(1)).min(8) as f64;
        let total_before = (g.claims_total.saturating_sub(lp.n)).min(12) as f64;
        let claimer_earlier = g.claims[c].saturating_sub(lp.n) as f64;
        let n_alive = g.n_alive() as f64;
        let holders = (0..g.n).filter(|&q| g.active(q)).count() as f64;
        let pulls: u32 = (0..g.n).map(|q| g.shots[q] as u32).sum();
        let creal = g.hm[p] as f64;
        let excess = (g.claims_total as i32 + g.hm[p] as i32 - g.truths() as i32).max(0) as f64;
        // Order must match research/fit_population.py FEATS.
        let x: [f64; FIT_CALL_NF] = [
            n, left, (left == 0.0) as u8 as f64, (left <= 2.0) as u8 as f64, plays_before, total_before, claimer_earlier, n_alive, holders,
            g.hand(p) as f64, g.shots[p] as f64, g.shots[c] as f64, (plays_before == 0.0) as u8 as f64,
            (lp.n >= 3) as u8 as f64, (n_alive == 2.0) as u8 as f64, (1 + pulls).min(10) as f64,
            creal, (creal == 0.0) as u8 as f64, excess, rate_trait,
        ];
        let mut z = FIT_CALL_B;
        for j in 0..FIT_CALL_NF {
            z += FIT_CALL_W[j] * (x[j] - FIT_CALL_MU[j]) / FIT_CALL_SD[j];
        }
        if g.net_seat[c] {
            z += fit_bot_tax();
        }
        1.0 / (1.0 + (-z).exp())
    }
}

impl Agent for FitHuman {
    fn decide_deal(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        deal_heuristic(g, p, deal_scale(), rng)
    }

    fn decide_call(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        let mut pm = FitHuman::call_prob(g, p, self.rate_trait);
        if self.read > 0.0 {
            if let Some(l) = g.last {
                if g.net_seat[l.player] {
                    let lie = !l.devil && l.n_true < l.n;
                    let z = (pm / (1.0 - pm).max(1e-9)).ln() + if lie { self.read } else { -self.read };
                    pm = 1.0 / (1.0 + (-z).exp());
                }
            }
        }
        let pr = (1.0 - self.blind) * pm + self.blind * self.rate_trait;
        rng.f64() < pr
    }

    fn decide_play(&mut self, g: &Game, p: usize, rng: &mut Rng) -> (u8, u8) {
        use crate::fit_data::*;
        let (m, nm, d) = (g.hm[p], g.hn[p], g.hd[p]);
        let opener = (g.claims_total == 0) as usize;
        let two = (g.n_alive() == 2) as usize;
        let key = (((m.min(5) as usize * 6 + nm.min(5) as usize) * 2 + d.min(1) as usize) * 2 + opener) * 2 + two;
        let base = &FIT_PLAY[key];
        let boost = self.bluff_z.exp() as f32;
        let mut pr = [0f32; FIT_NA];
        let mut tot = 0f32;
        for i in 0..FIT_NA {
            let (t, b) = FIT_ACTS[i];
            let legal = if i == 0 { d > 0 } else { t <= m && b <= nm };
            if legal {
                pr[i] = base[i] * if i > 0 && b > 0 { boost } else { 1.0 };
                tot += pr[i];
            }
        }
        if tot <= 0.0 {
            // Table empty for this key (cannot happen for reachable hands): any legal play.
            if m + nm == 0 { return DEVIL_PLAY; }
            return if m > 0 { (1, 0) } else { (0, 1) };
        }
        for v in pr.iter_mut() { *v /= tot; }
        let a = rng.sample(&pr, FIT_NA);
        if a == 0 { DEVIL_PLAY } else { FIT_ACTS[a] }
    }
}


/// The live advisor's wrapper around a network, for the simulator (2026-11-06): the v54-v59
/// ordering rules (junk first, two at a time; real cards thrown together once the junk is gone;
/// dump up to 3 junk when someone is down to 2 cards; one opener in four real, rising with
/// exposures), the call policy of the live bot (`nocall`: the network's own call decision is
/// never used, as under the lie-detector override; or a threshold on the network's call
/// probability), argmax instead of sampling, and the Devil's Deal gate (card count >= 25%).
pub struct RuleWrap {
    pub m: Arc<crate::nn::Model>,
    pub rules: bool,
    pub nocall: bool,
    pub callthresh: Option<f64>,
    pub greedy: bool,
    pub open_true: f64,
    /// Shared counters: plays, bluffs (any fake card), rule-forced plays.
    pub stats: Arc<[std::sync::atomic::AtomicU64; 3]>,
    /// v61: data-fitted call model instead of the network call head (prior = population rate,
    /// think time = fit mean; the simulator has neither names nor think times).
    pub datacall: Option<Arc<crate::datacall::DataCall>>,
}

impl RuleWrap {
    pub fn new(m: Arc<crate::nn::Model>, rules: bool, nocall: bool, callthresh: Option<f64>, greedy: bool, stats: Arc<[std::sync::atomic::AtomicU64; 3]>) -> RuleWrap {
        RuleWrap { m, rules, nocall, callthresh, greedy, open_true: 0.25, stats, datacall: None }
    }

    fn pick(&self, probs: &[f32], n: usize, rng: &mut Rng) -> usize {
        if self.greedy {
            (0..n).max_by(|&a, &b| probs[a].partial_cmp(&probs[b]).unwrap()).unwrap_or(0)
        } else {
            rng.sample(probs, n)
        }
    }
}

impl Agent for RuleWrap {
    fn decide_deal(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        if deal_p_all_lies(g, p) < 0.25 {
            return false;
        }
        let out = self.m.logits(g, p, crate::nn::KIND_DEAL);
        let pr = crate::nn::Net::probs(&out, 0, 2);
        if self.greedy { pr[1] > 0.5 } else { rng.sample(&pr, 2) == 1 }
    }

    fn decide_call(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        if self.nocall {
            return false;
        }
        if let Some(dc) = &self.datacall {
            let lp = match g.last { Some(lp) => lp, None => return false };
            if let Some(f) = crate::datacall::features(g, p, dc.mean_think(), dc.p0, g.rec_called[lp.player], g.rec_lies[lp.player]) {
                let t = self.callthresh.unwrap_or(dc.thresh);
                return dc.p_lie(&f) >= t;
            }
            return false;
        }
        let out = self.m.logits(g, p, crate::nn::KIND_CALL);
        let pr = crate::nn::Net::probs(&out, 2, 2);
        match self.callthresh {
            Some(t) => pr[1] as f64 > t,
            None => if self.greedy { pr[1] > 0.5 } else { rng.sample(&pr, 2) == 1 },
        }
    }

    fn decide_play(&mut self, g: &Game, p: usize, rng: &mut Rng) -> (u8, u8) {
        use std::sync::atomic::Ordering::Relaxed;
        let lg = &LEGAL[g.hm[p] as usize][g.hn[p] as usize];
        let na = g.n_actions(p);
        let act = |i: usize| if i < lg.n { lg.acts[i] } else { DEVIL_PLAY };
        let (real, junk) = (g.hm[p], g.hn[p]);
        let mut chosen: Option<(u8, u8)> = None;
        if self.rules {
            let my_plays = g.rd_n[p];
            let my_cards = g.hand(p);
            let min_other = (0..g.n).filter(|&q| q != p && g.active(q)).map(|q| g.hand(q)).min();
            let threat = matches!(min_other, Some(m) if m <= 2 && my_cards >= m + 2);
            let open_true = (self.open_true + 0.20 * g.rec_lies[p] as f64).min(0.65);
            let want: Option<(u8, u8)> = if real > 0 && junk > 0 {
                if threat {
                    Some((0, junk.min(3).max(1)))
                } else if my_plays == 0 && real >= 2 && rng.f64() < open_true {
                    Some((1, 0))
                } else {
                    Some((0, junk.min(2).max(1)))
                }
            } else if real > 0 && junk == 0 {
                Some((real.min(3), 0))
            } else {
                None
            };
            if let Some(w) = want {
                let mut pick = (0..na).find(|&j| act(j) == w);
                if pick.is_none() && w.1 > 2 { pick = (0..na).find(|&j| act(j) == (0, 2)); }
                if pick.is_none() && w.1 > 1 { pick = (0..na).find(|&j| act(j) == (0, 1)); }
                if pick.is_none() && w.0 > 1 { pick = (0..na).find(|&j| act(j) == (w.0 - 1, 0)); }
                if pick.is_none() && w.0 > 2 { pick = (0..na).find(|&j| act(j) == (1, 0)); }
                if let Some(j) = pick {
                    chosen = Some(act(j));
                    self.stats[2].fetch_add(1, Relaxed);
                }
            }
        }
        let a = match chosen {
            Some(a) => a,
            None => {
                let out = self.m.logits(g, p, crate::nn::KIND_PLAY);
                let mut probs = crate::nn::Net::probs(&out, 4, na);
                // The advisor folds mixed plays into the pure bluff of the same size when possible.
                for i in 0..na {
                    let a = act(i);
                    if a != DEVIL_PLAY && a.0 > 0 && a.1 > 0 && g.hn[p] >= a.0 + a.1 {
                        if let Some(j) = (0..na).find(|&j| act(j) == (0, a.0 + a.1)) {
                            probs[j] += probs[i];
                            probs[i] = 0.0;
                        }
                    }
                }
                let i = self.pick(&probs, na, rng);
                act(i)
            }
        };
        // v63 candidate (LIARS_TRUTH_FIRST=1): same throw size, real cards instead of fakes when the hand allows it.
        let a = if truth_first() && a != DEVIL_PLAY && a.1 > 0 && g.hm[p] >= a.0 + a.1 {
            let want = (a.0 + a.1, 0u8);
            if (0..na).any(|j| act(j) == want) { want } else { a }
        } else { a };
        self.stats[0].fetch_add(1, Relaxed);
        if a != DEVIL_PLAY && a.1 > 0 {
            self.stats[1].fetch_add(1, Relaxed);
        }
        a
    }
}

/// LIARS_TRUTH_FIRST=1: throw matching cards instead of fakes whenever the chosen size fits the matching cards held.
pub fn truth_first() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| std::env::var("LIARS_TRUTH_FIRST").map(|v| v == "1").unwrap_or(false))
}

pub fn make_agent(kind: &str, policy: Option<&Arc<Policy>>) -> Box<dyn Agent> {
    match kind {
        "prob" => Box::new(ProbAgent::default()),
        "honest" => Box::new(HonestAgent),
        "random" => Box::new(RandomAgent { call_rate: 0.3 }),
        "human" => Box::new(HumanAgent::default()),
        // Fast dumper (2 then 3 or 3 then 2), an average caller otherwise.
        "dumper" => Box::new(DumperAgent::new(HumanAgent::default(), 0.5, 0.5)),
        // Jaime-style reader: keeps calling players caught bluffing.
        "reader" => Box::new(ReaderAgent { inner: HumanAgent::default(), grudge: 0.3 }),
        // Munk-style dumper: real cards first, so the hand-emptying throw is usually a lie.
        "dumpliar" => Box::new(DumperAgent { liar: true, ..DumperAgent::new(HumanAgent::default(), 0.6, 0.5) }),
        // Grand_Master-style: calls about twice as often as the average logged player.
        "heavy" => Box::new(HumanAgent { call_scale: 2.2, ..HumanAgent::default() }),
        "ranked" => Box::new(HumanAgent { ranked: true, ..HumanAgent::default() }),
        // 2026-11-06: a ranked human fitted to the logs (research/fit_population.py); traits drawn per player.
        "rankedfit" | "rankedfit3" | "rankedfit1v1" => Box::new(FitHuman::random_global()),
        // Logged archetypes: suffle / A-胖胖 style honest big-claimers, and heavy bluffers.
        "honestbig" => Box::new(HumanAgent { call_scale: 1.0, bluff_rate: 0.07, big: 0.7, bluff2: 0.2, bluff3: 0.02, save: 0.6, open_call: 1.0, ranked: false, rate_trait: 0.20 }),
        "bluffer" => Box::new(HumanAgent { call_scale: 1.0, bluff_rate: 0.45, big: 0.35, bluff2: 0.5, bluff3: 0.2, save: 0.3, open_call: 1.0, ranked: false, rate_trait: 0.20 }),
        // 2026-11-01: a ranked player who calls like the real top decile.
        "rankedheavy" => Box::new(HumanAgent { ranked: true, rate_trait: 0.40, ..HumanAgent::default() }),
        // WeirderLime-style: calls almost half his chances, openers included.
        "caller" => Box::new(HumanAgent { call_scale: 2.8, open_call: 5.0, ..HumanAgent::default() }),
        // Styles never seen in training, to test whether reading players generalizes.
        "aggro" => Box::new(ProbAgent { base_threshold: 0.35, risk_weight: 0.2, bluff_rate: 0.45 }),
        "timid" => Box::new(ProbAgent { base_threshold: 0.75, risk_weight: 0.6, bluff_rate: 0.02 }),
        "policy" | "self" | "cfr" => Box::new(CfrAgent::new(policy.expect("policy required").clone())),
        "nn" => Box::new(crate::nn::NnAgent { net: crate::nn::NET.get().expect("--net required").clone() }),
        other => panic!("unknown agent kind {other}"),
    }
}

pub fn opponent_lineup(spec: &str, n: usize) -> Vec<&str> {
    if spec == "mixed" || spec == "newmix" {
        let base = if spec == "mixed" { ["prob", "honest", "random"] } else { ["aggro", "timid", "honest"] };
        (0..n).map(|i| base[i % base.len()]).collect()
    } else {
        vec![spec; n]
    }
}

/// Plays one full game; `on_round` sees each round's opening state before any move.
pub fn play_game(
    agents: &mut [Box<dyn Agent>],
    rng: &mut Rng,
    rule: StarterRule,
    devil: bool,
    mut on_round: impl FnMut(&Game),
) -> usize {
    let mut g = Game::new_game(agents.len(), devil, rng);
    on_round(&g);
    let mut steps = 0;
    while !g.game_over {
        steps += 1;
        assert!(steps < 100_000, "game exceeded step limit");
        let p = g.current;
        if g.deal_legal() && agents[p].decide_deal(&g, p, rng) {
            g.apply_deal(rng);
        } else if g.is_forced(p) {
            g.apply_call();
        } else if g.last.is_some() && agents[p].decide_call(&g, p, rng) {
            g.apply_call();
        } else {
            let a = agents[p].decide_play(&g, p, rng);
            g.apply_action(a);
        }
        if g.round_over {
            g.finish_round(rule, rng);
            if !g.game_over {
                on_round(&g);
            }
        }
    }
    g.winner
}

/// Win rate of `hero` against `opponents`, rotating the hero's seat. Policy opponents ("self")
/// use `opp_policy` when given, for head-to-head comparisons between two policies. Returns (win rate, miss rate).
pub fn evaluate(
    hero: &str,
    policy: Option<&Arc<Policy>>,
    opp_policy: Option<&Arc<Policy>>,
    opponents: &str,
    games: usize,
    n_players: usize,
    seed: u64,
    rule: StarterRule,
    devil: bool,
) -> (f64, Option<f64>) {
    let mut rng = Rng::new(seed);
    let lineup = opponent_lineup(opponents, n_players - 1);
    let mut hero_agent = Some(make_agent(hero, policy));
    let mut opps: Vec<Box<dyn Agent>> = lineup.iter().map(|k| make_agent(k, opp_policy.or(policy))).collect();
    let mut wins = 0;
    for gi in 0..games {
        let seat = gi % n_players;
        let mut agents: Vec<Box<dyn Agent>> = Vec::with_capacity(n_players);
        let mut it = opps.drain(..);
        for s in 0..n_players {
            if s == seat {
                agents.push(hero_agent.take().unwrap());
            } else {
                agents.push(it.next().unwrap());
            }
        }
        drop(it);
        if play_game(&mut agents, &mut rng, rule, devil, |_| {}) == seat {
            wins += 1;
        }
        for (s, a) in agents.into_iter().enumerate() {
            if s == seat {
                hero_agent = Some(a);
            } else {
                opps.push(a);
            }
        }
    }
    let miss = hero_agent
        .unwrap()
        .miss_stats()
        .and_then(|(m, l)| if l > 0 { Some(m as f64 / l as f64) } else { None });
    (wins as f64 / games as f64, miss)
}

#[cfg(test)]
mod deal_tests {
    use super::*;

    /// Devil's Deal happens sometimes among bots, and wins a believable share of the time.
    #[test]
    fn deal_rates() {
        let mut rng = Rng::new(7);
        let mut agents: Vec<Box<dyn Agent>> = vec![
            Box::new(HumanAgent::default()), Box::new(ProbAgent::default()),
            Box::new(HonestAgent), Box::new(RandomAgent { call_rate: 0.3 }),
        ];
        let (mut rounds, mut deals, mut wins) = (0, 0, 0);
        for _ in 0..3000 {
            let mut g = Game::new_game(4, true, &mut rng);
            while !g.game_over {
                let p = g.current;
                if g.deal_legal() && agents[p].decide_deal(&g, p, &mut rng) {
                    g.apply_deal(&mut rng);
                } else if g.is_forced(p) {
                    g.apply_call();
                } else if g.last.is_some() && agents[p].decide_call(&g, p, &mut rng) {
                    g.apply_call();
                } else {
                    let a = agents[p].decide_play(&g, p, &mut rng);
                    g.apply_action(a);
                }
                if g.round_over {
                    rounds += 1;
                    deals += g.deal_called as u32;
                    wins += (g.deal_called && g.lie) as u32;
                    g.finish_round(StarterRule::Loser, &mut rng);
                }
            }
        }
        println!("rounds {rounds} deals {deals} ({:.1}%) won {wins}", 100.0 * deals as f64 / rounds as f64);
        assert!(deals > 0 && wins > 0 && wins < deals);
    }
}

#[cfg(test)]
mod traits_tests {
    use super::*;

    /// Averaged over random trait draws, claim sizes should stay near the logged 45/36/19.
    #[test]
    fn trait_mix_sizes() {
        let mut rng = Rng::new(5);
        let (mut sizes, mut lies) = ([0u32; 4], 0u32);
        let n = 40000;
        for _ in 0..n {
            let mut agent = HumanAgent::random_traits(1.0, &mut rng);
            let g = Game::new_game(4, false, &mut rng);
            let a = agent.decide_play(&g, g.current, &mut rng);
            sizes[(a.0 + a.1) as usize] += 1;
            lies += (a.1 > 0) as u32;
        }
        let t = n as f64 / 100.0;
        println!("trait mix sizes 1/2/3: {:.0}% {:.0}% {:.0}%  bluffs {:.0}%",
            sizes[1] as f64 / t, sizes[2] as f64 / t, sizes[3] as f64 / t, lies as f64 / t);
    }
}

#[cfg(test)]
mod call_diag {
    use super::*;

    /// How often a policy calls when it holds no real cards vs some, against random-trait humans,
    /// and how often the claim it faces is actually a lie. POLICY=path cargo test call_diag -- --nocapture
    #[test]
    fn call_diag() {
        let Ok(path) = std::env::var("POLICY") else { return };
        let pol = Arc::new(Policy::load(&path).expect("policy"));
        let mut rng = Rng::new(11);
        // [no real, some real] x (decisions, summed call prob, claims that were lies)
        let mut st = [[0f64; 3]; 2];
        for gi in 0..20000 {
            let mut agents: Vec<Box<dyn Agent>> = (0..4)
                .map(|_| Box::new(HumanAgent::random_traits(0.4 + 2.1 * rng.f64() * rng.f64(), &mut rng)) as Box<dyn Agent>)
                .collect();
            let hero = gi % 4;
            let mut g = Game::new_game(4, false, &mut rng);
            while !g.game_over {
                let p = g.current;
                if p == hero && g.last.is_some() && !g.is_forced(p) && g.alive[p] {
                    if let Some(pr) = pol.call_probs(call_key(&g, p, &pol.feat())) {
                        let lp = g.last.unwrap();
                        let row = &mut st[(g.hm[p] > 0) as usize];
                        row[0] += 1.0;
                        row[1] += pr[1] as f64;
                        row[2] += (lp.n_true < lp.n) as u8 as f64;
                    }
                }
                let mut cfr = CfrAgent::new(pol.clone());
                let a: &mut dyn Agent = if p == hero { &mut cfr } else { agents[p].as_mut() };
                if g.is_forced(p) {
                    g.apply_call();
                } else if g.last.is_some() && a.decide_call(&g, p, &mut rng) {
                    g.apply_call();
                } else {
                    let act = a.decide_play(&g, p, &mut rng);
                    g.apply_action(act);
                }
                if g.round_over {
                    g.finish_round(StarterRule::Loser, &mut rng);
                }
            }
        }
        for (i, name) in ["no real cards  ", "some real cards"].iter().enumerate() {
            let [n, c, l] = st[i];
            println!("{name}: decisions {n:.0}, calls {:.0}% of the time, claim was a lie {:.0}%", 100.0 * c / n, 100.0 * l / n);
        }
    }
}

#[cfg(test)]
mod lie_by_left {
    use super::*;

    /// Lie rate of called claims by the claimer's cards left, among random-trait humans
    /// (to compare with logged lobbies: 3-4 left 47%, 1-2 left 25%, emptied 25%).
    #[test]
    fn lie_by_left() {
        let mut rng = Rng::new(21);
        let mut st = [[0u32; 2]; 3];
        for _ in 0..30000 {
            let mut agents: Vec<HumanAgent> = (0..4).map(|_| HumanAgent::random_traits(0.4 + 2.1 * rng.f64() * rng.f64(), &mut rng)).collect();
            let mut g = Game::new_game(4, false, &mut rng);
            while !g.game_over {
                let p = g.current;
                if g.is_forced(p) {
                    g.apply_call();
                } else if g.last.is_some() && agents[p].decide_call(&g, p, &mut rng) {
                    let lp = g.last.unwrap();
                    let left = g.hand(lp.player);
                    let b = if left == 0 { 2 } else if left <= 2 { 1 } else { 0 };
                    st[b][0] += 1;
                    st[b][1] += (lp.n_true < lp.n) as u32;
                    g.apply_call();
                } else {
                    let a = agents[p].decide_play(&g, p, &mut rng);
                    g.apply_action(a);
                }
                if g.round_over {
                    g.finish_round(StarterRule::Loser, &mut rng);
                }
            }
        }
        for (i, name) in ["3-4 left", "1-2 left", "emptied "].iter().enumerate() {
            println!("{name}: {}/{} = {:.0}% lies", st[i][1], st[i][0], 100.0 * st[i][1] as f64 / st[i][0] as f64);
        }
    }
}

#[cfg(test)]
mod late_diag {
    use super::*;

    /// Every claim (called or not) that leaves 1-2 cards: share that were lies, and of those
    /// lies, how many came from a player who had no real cards left.
    #[test]
    fn late_diag() {
        let mut rng = Rng::new(31);
        let (mut claims, mut lies, mut forced_lies) = (0u32, 0u32, 0u32);
        for _ in 0..20000 {
            let mut agents: Vec<HumanAgent> = (0..4).map(|_| HumanAgent::random_traits(0.4 + 2.1 * rng.f64() * rng.f64(), &mut rng)).collect();
            let mut g = Game::new_game(4, false, &mut rng);
            while !g.game_over {
                let p = g.current;
                if g.is_forced(p) {
                    g.apply_call();
                } else if g.last.is_some() && agents[p].decide_call(&g, p, &mut rng) {
                    g.apply_call();
                } else {
                    let had_real = g.hm[p] > 0;
                    let a = agents[p].decide_play(&g, p, &mut rng);
                    g.apply_action(a);
                    let left = g.hand(p);
                    if (1..=2).contains(&left) {
                        claims += 1;
                        if a.1 > 0 {
                            lies += 1;
                            forced_lies += !had_real as u32;
                        }
                    }
                }
                if g.round_over {
                    g.finish_round(StarterRule::Loser, &mut rng);
                }
            }
        }
        println!("claims leaving 1-2 cards: {claims}, lies {:.0}%, of the lies {:.0}% had no real card", 100.0 * lies as f64 / claims as f64, 100.0 * forced_lies as f64 / lies as f64);
    }
}

#[cfg(test)]
mod deal_h2h {
    use super::*;

    /// One HERO policy vs three OPP policies in Devil mode: wins, and each side's Devil's Deal use.
    #[test]
    fn deal_h2h() {
        let (Ok(hp), Ok(op)) = (std::env::var("HERO"), std::env::var("OPP")) else { return };
        let hero_pol = Arc::new(Policy::load(&hp).unwrap());
        let opp_pol = Arc::new(Policy::load(&op).unwrap());
        let mut rng = Rng::new(5);
        let (mut wins, games) = (0u32, 4000u32);
        // [hero, opp]: deals called, deals won, liar calls, liar calls right
        let mut st = [[0u32; 4]; 2];
        let mut pulls = [0u32; 4];
        for gi in 0..games {
            let hero = (gi % 4) as usize;
            let mut agents: Vec<CfrAgent> = (0..4).map(|q| CfrAgent::new(if q == hero { hero_pol.clone() } else { opp_pol.clone() })).collect();
            let mut g = Game::new_game(4, true, &mut rng);
            while !g.game_over {
                let p = g.current;
                let side = (p != hero) as usize;
                let no_deal = side == 0 && std::env::var("NODEAL").is_ok();
                if g.deal_legal() && !no_deal && agents[p].decide_deal(&g, p, &mut rng) {
                    g.apply_deal(&mut rng);
                    st[side][0] += 1;
                    st[side][1] += g.lie as u32;
                } else if g.is_forced(p) {
                    g.apply_call();
                } else if g.last.is_some() && agents[p].decide_call(&g, p, &mut rng) {
                    g.apply_call();
                    st[side][2] += 1;
                    st[side][3] += g.lie as u32;
                } else {
                    let a = agents[p].decide_play(&g, p, &mut rng);
                    g.apply_action(a);
                }
                if g.round_over {
                    // Why the hero pulls: 0 called a true claim, 1 caught lying, 2 Devil reveal, 3 deal.
                    if g.pullers[hero] {
                        let why = if g.deal_called { 3 } else if g.devil_reveal { 2 } else if g.loser == hero && g.accused == hero { 1 } else { 0 };
                        pulls[why] += 1;
                    }
                    g.finish_round(StarterRule::Loser, &mut rng);
                }
            }
            wins += (g.winner == hero) as u32;
        }
        println!("hero win rate {:.3}", wins as f64 / games as f64);
        println!("hero pulls: called a true claim {}, caught lying {}, Devil reveal {}, deal {}", pulls[0], pulls[1], pulls[2], pulls[3]);
        for (i, name) in ["hero (x1)", "opps (x3)"].iter().enumerate() {
            let s = st[i];
            println!("{name}: deals {} won {} | liar calls {} right {}", s[0], s[1], s[2], s[3]);
        }
    }
}
