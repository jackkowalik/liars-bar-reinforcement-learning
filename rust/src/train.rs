use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::engine::{undominate, Game, StarterRule, LEGAL, MAX_ACTIONS, MAX_PLAYERS};

use crate::policy::{call_key, deal_key, play_key, Feat, Table, Tables};
use crate::agents::{Agent, HonestAgent, HumanAgent, ProbAgent, RandomAgent};
use crate::rng::Rng;
use crate::value::{StartSampler, ValueModel};

#[derive(Clone, Copy)]
pub struct Rec {
    table: u8,
    slot: u32,
    n: u8,
    a: u8,
    sig: [f32; MAX_ACTIONS],
}

/// Outcome-sampling Monte Carlo CFR with regret matching+. Each round pays the change in the
/// traverser's win probability, so chamber asymmetry is part of the payoff.
pub struct Trainer {
    pub tables: Tables,
    pub t: AtomicU64,
    pub n_players: usize,
    pub epsilon: f32,
    pub rule: StarterRule,
    pub value: ValueModel,
    pub starts: StartSampler,
    /// Chance each non-traverser seat is a fixed-style bot for the round instead of the
    /// policy, so the policy can't assume every opponent bluffs the way it does.
    pub bot_share: f32,
    /// Share of episodes that start from a fresh game instead of a sampled mid-game state.
    pub full_game_share: f32,
    /// Use per-player history buckets in info sets (false = ablation).
    pub history: bool,
    /// Re-draw bot styles every episode.
    pub vary_bots: bool,
    /// Devil mode rules.
    pub devil: bool,
    /// Info-set feature switches (ablation).
    pub feat: Feat,
}

/// Per-thread opponent bots for population training.
pub struct Bots {
    honest: HonestAgent,
    prob: ProbAgent,
    random: RandomAgent,
    /// One human-like player per seat, each with its own archetype.
    human: [HumanAgent; MAX_PLAYERS],
    /// Include the human-like caller in the opponent pool.
    pub with_human: bool,
}

impl Default for Bots {
    fn default() -> Self {
        Bots { honest: HonestAgent, prob: ProbAgent::default(), random: RandomAgent { call_rate: 0.3 }, human: std::array::from_fn(|_| HumanAgent::default()), with_human: false }
    }
}

impl Bots {
    /// New styles each episode, so history buckets have to mean "this player tends to lie"
    /// rather than "this is the random bot".
    fn randomize(&mut self, rng: &mut Rng) {
        self.prob = ProbAgent {
            base_threshold: 0.3 + 0.5 * rng.f64(),
            risk_weight: 0.6 * rng.f64(),
            bluff_rate: 0.5 * rng.f32(),
        };
        self.random = RandomAgent { call_rate: 0.1 + 0.4 * rng.f32() };
        // Wide spread: very passive players up to Grand_Master-style heavy callers.
        for h in self.human.iter_mut() {
            let call_scale = 0.4 + 2.1 * rng.f64() * rng.f64();
            *h = HumanAgent::random_traits(call_scale, rng);
        }
    }

    fn get(&mut self, kind: u8, seat: usize) -> &mut dyn Agent {
        match kind {
            1 => &mut self.honest,
            2 => &mut self.prob,
            4 => &mut self.human[seat],
            _ => &mut self.random,
        }
    }
}

impl Trainer {
    #[inline]
    fn table(&self, id: u8) -> &Table {
        if id == 0 { &self.tables.call } else { &self.tables.play }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn act(&self, id: u8, key: u64, n: usize, traverser: bool, t: u64, rng: &mut Rng, recs: &mut Vec<Rec>) -> (usize, f64) {
        let tbl = self.table(id);
        let mut sig = [0.0f32; MAX_ACTIONS];
        let slot = tbl.slot(key);
        tbl.sigma(slot, n, &mut sig);
        if traverser {
            let eps = self.epsilon;
            let mut probs = [0.0f32; MAX_ACTIONS];
            for k in 0..n {
                probs[k] = eps / n as f32 + (1.0 - eps) * sig[k];
            }
            let a = rng.sample(&probs, n);
            recs.push(Rec { table: id, slot: slot as u32, n: n as u8, a: a as u8, sig });
            (a, probs[a] as f64)
        } else {
            let a = rng.sample(&sig, n);
            tbl.add_strat(slot, n, &sig, t as f64);
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
            let tbl = self.table(rec.table);
            let slot = rec.slot as usize;
            let a = rec.a as usize;
            let sa = rec.sig[a];
            for b in 0..rec.n as usize {
                let r = tbl.r(slot, b);
                let v = if b == a { r + w * (1.0 - sa) } else { r - w * sa };
                tbl.set_r(slot, b, if v > 0.0 { v } else { 0.0 });
            }
            tail *= sa as f64;
        }
    }

    /// One episode: a fresh game (`full_game_share` of the time) or a sampled mid-game state,
    /// played to the end of the game so per-player records build up as they do in real play.
    /// Every round is one iteration: a random alive policy seat traverses it and is paid the
    /// round's change in its win probability. Seat styles are fixed for the whole episode.
    pub fn iterate(&self, rng: &mut Rng, recs: &mut Vec<Rec>, bots: &mut Bots) {
        let n = self.n_players;
        let (alive, shots, starter) = if rng.f32() < self.full_game_share {
            let mut a = [false; MAX_PLAYERS];
            a[..n].iter_mut().for_each(|x| *x = true);
            (a, [0u8; MAX_PLAYERS], rng.below(n as u32) as usize)
        } else {
            self.starts.sample(rng)
        };
        let mut g = Game::new_mode(n, alive, shots, starter, self.devil, rng);
        if self.vary_bots {
            bots.randomize(rng);
        }
        // 0 = policy, 1 = honest, 2 = prob, 3 = random; at least one alive policy seat.
        let mut seat = [0u8; MAX_PLAYERS];
        for k in seat.iter_mut().take(n) {
            if rng.f32() < self.bot_share {
                // Bot styles: 1 honest, 2 prob, 3 random, 4 human-like (when enabled), equally likely.
                // Weighting human-like double starved training of passive opponents.
                *k = 1 + rng.below(if bots.with_human { 4 } else { 3 }) as u8;
            }
        }
        let pick = rng.below(g.n_alive() as u32) as usize;
        seat[(0..n).filter(|&q| alive[q]).nth(pick).unwrap()] = 0;
        loop {
            let mine: Vec<usize> = (0..n).filter(|&q| g.alive[q] && seat[q] == 0).collect();
            if mine.is_empty() {
                return;
            }
            let traverser = mine[rng.below(mine.len() as u32) as usize];
            let t = self.t.fetch_add(1, Relaxed) + 1;
            self.play_round(&mut g, &seat, traverser, t, rng, recs, bots);
            g.finish_round(self.rule, rng);
            if g.game_over {
                return;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn play_round(&self, g: &mut Game, seat: &[u8; MAX_PLAYERS], traverser: usize, t: u64, rng: &mut Rng, recs: &mut Vec<Rec>, bots: &mut Bots) {
        let baseline = self.value.state_value(&g.alive, &g.shots, g.starter, traverser);
        recs.clear();
        let mut q = 1.0f64;
        while !g.round_over {
            let p = g.current;
            if g.deal_legal() {
                let deal = if seat[p] != 0 {
                    bots.get(seat[p], p).decide_deal(g, p, rng)
                } else {
                    let (a, s) = self.act(0, deal_key(g, p), 2, p == traverser, t, rng, recs);
                    q *= s;
                    a == 1
                };
                if deal {
                    g.apply_deal(rng);
                    continue;
                }
            }
            if g.is_forced(p) {
                g.apply_call();
                continue;
            }
            if seat[p] != 0 {
                let bot = bots.get(seat[p], p);
                if g.last.is_some() && bot.decide_call(g, p, rng) {
                    g.apply_call();
                } else {
                    let a = bot.decide_play(g, p, rng);
                    g.apply_action(a);
                }
                continue;
            }
            if g.last.is_some() {
                let (a, s) = self.act(0, call_key(g, p, &self.feat), 2, p == traverser, t, rng, recs);
                q *= s;
                if a == 1 {
                    g.apply_call();
                    continue;
                }
            }
            let lg = &LEGAL[g.hm[p] as usize][g.hn[p] as usize];
            let (a, s) = self.act(1, play_key(g, p, &self.feat), g.n_actions(p), p == traverser, t, rng, recs);
            q *= s;
            if a < lg.n {
                let (tt, b) = undominate(g.hn[p], lg.acts[a]);
                g.apply_play(tt, b);
            } else {
                g.apply_devil();
            }
        }
        let v = self.value.after_round(g, traverser, self.rule);
        self.update(recs, v - baseline, q);
    }
}
