//! Python bridge: thousands of Liar's Deck games stepped side by side, so a network on the GPU
//! can answer every pending decision in one batch.
//!
//! The game engine, opponents, champion table policy, encoder and win-probability model are the
//! same source files the trainer and advisor use (included below by path).
#![allow(dead_code)]

#[path = "../../rust/src/agents.rs"]
mod agents;
#[path = "../../rust/src/engine.rs"]
mod engine;
#[path = "../../rust/src/fit_data.rs"]
mod fit_data;
#[path = "../../rust/src/nn.rs"]
mod nn;
#[path = "../../rust/src/policy.rs"]
mod policy;
#[path = "../../rust/src/rng.rs"]
mod rng;
#[path = "../../rust/src/value.rs"]
mod value;

use std::sync::Arc;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

use agents::{make_agent, Agent, DumperAgent, FitHuman, HumanAgent, ProbAgent, ReaderAgent};
use engine::{undominate, Game, StarterRule, DEVIL_PLAY, LEGAL, MAX_ACTIONS, MAX_PLAYERS};
use nn::{encode, encode_hist, EV_DIM, HIST_LEN, KIND_CALL, KIND_DEAL, KIND_PLAY, N_FEAT};
use policy::Policy;
use rng::Rng;
use value::ValueModel;

const RULE: StarterRule = StarterRule::Loser;

/// One game and the bookkeeping around its network-controlled seats.
struct Slot {
    g: Game,
    /// Seats played by the network (the rest are bots), and which of those are a frozen past
    /// version (league opponents: their decisions are not trained on).
    net: [bool; MAX_PLAYERS],
    snap: [bool; MAX_PLAYERS],
    bots: Vec<Option<Box<dyn Agent>>>,
    /// Where the current network turn is: 0 Devil's Deal question, 1 call question, 2 play.
    stage: u8,
    /// Pending network decision: seat, kind, decision id.
    pending: Option<(usize, u8, u64)>,
    /// Win probability of each network seat at round start, and their decisions this round.
    base: [f64; MAX_PLAYERS],
    round_ids: Vec<(usize, u64)>,
}

#[pyclass(unsendable)]
struct VecEnv {
    slots: Vec<Slot>,
    rng: Rng,
    value: ValueModel,
    champ: Option<Arc<Policy>>,
    devil: bool,
    /// Opponent mix for training seats: human-like, champion, probability bot (random settings),
    /// honest bot, frozen past network; the rest is the learning network.
    mix: (f32, f32, f32, f32, f32),
    /// Evaluation: seat 0 is the network, every other seat is this bot kind.
    eval_opp: Option<String>,
    next_id: u64,
    rewards: Vec<(u64, f32)>,
    games: u64,
    net_wins: f64,
}

impl VecEnv {
    fn new_slot(&mut self) -> Slot {
        let n = 4;
        // Devil training: some games start short-handed (3 left, or 1v1), mostly with a fast
        // dumper among the survivors, the situations the advisor defends worst.
        let short = match self.eval_opp.as_deref() {
            Some("dumper3") => 3,
            Some("dumper1v1") | Some("ranked1v1") | Some("rankedfit1v1") => 2,
            Some("ranked3") | Some("rankedfit3") => 3,
            Some(_) => 4,
            None if self.devil => {
                let r = self.rng.f32();
                if r < 0.15 { 3 } else if r < 0.3 { 2 } else { 4 }
            }
            None => 4,
        };
        let mut g = if short < 4 {
            let mut alive = [false; MAX_PLAYERS];
            alive[0] = true;
            let mut others = vec![1usize, 2, 3];
            for _ in 1..short {
                let i = self.rng.below(others.len() as u32) as usize;
                alive[others.swap_remove(i)] = true;
            }
            let mut shots = [0u8; MAX_PLAYERS];
            for q in 0..n {
                if alive[q] {
                    shots[q] = self.rng.below(4) as u8;
                }
            }
            let live: Vec<usize> = (0..n).filter(|&q| alive[q]).collect();
            let starter = live[self.rng.below(live.len() as u32) as usize];
            Game::new_mode(n, alive, shots, starter, self.devil, &mut self.rng)
        } else {
            Game::new_game(n, self.devil, &mut self.rng)
        };
        // Keep the public event history for the transformer (the first round start by hand).
        g.record = true;
        // Devil games are ranked lobbies half the time (the "ranked" input is on and every
        // human-like seat plays the ranked fit); the other half keep the old mix, unchanged.
        g.ranked = self.devil && match self.eval_opp.as_deref() {
            Some(k) => k.starts_with("ranked"),
            None => self.rng.f32() < 0.5,
        };
        let na = g.n_alive() as u8;
        g.log_ev(engine::Ev { kind: 5, actor: g.starter as u8, n: na, ..Default::default() });
        let mut net = [false; MAX_PLAYERS];
        let mut snap = [false; MAX_PLAYERS];
        let mut bots: Vec<Option<Box<dyn Agent>>> = (0..n).map(|_| None).collect();
        match self.eval_opp.clone() {
            Some(kind) => {
                net[0] = true;
                for (q, b) in bots.iter_mut().enumerate().skip(1) {
                    let _ = q;
                    *b = Some(match kind.as_str() {
                        "champ" => make_agent("policy", self.champ.as_ref()),
                        "dumper3" | "dumper1v1" => Box::new(DumperAgent::random(&mut self.rng)),
                        "ranked1v1" | "ranked3" => make_agent("ranked", None),
                        "rankedfit" | "rankedfit3" | "rankedfit1v1" => Box::new(FitHuman::random(&mut self.rng)),
                        k => make_agent(k, self.champ.as_ref()),
                    });
                }
            }
            None => {
                for q in 0..n {
                    let r = self.rng.f32();
                    let (h, c, p, hon, sn) = self.mix;
                    bots[q] = if r < h || (g.ranked && r < h + c + p + hon) {
                        // A third of the human-like seats are heavy callers (openers included).
                        let u = self.rng.f32();
                        if g.ranked && u < 0.7 {
                            // Ranked lobby (2026-11-06): 70% of human-like seats are humans fitted to the logs.
                            Some(Box::new(FitHuman::random(&mut self.rng)))
                        } else if g.ranked {
                            // The rest keep the older ranked archetype for diversity.
                            let cs = 0.4 + 2.1 * self.rng.f64() * self.rng.f64();
                            Some(Box::new(HumanAgent { ranked: true, ..HumanAgent::random_traits(cs, &mut self.rng) }))
                        } else if u < 0.35 {
                            Some(Box::new(HumanAgent::heavy_caller(&mut self.rng)))
                        } else if self.devil && u < 0.5 {
                            Some(Box::new(DumperAgent::random(&mut self.rng)))
                        } else if u < 0.6 {
                            // About 10% of human-like seats: readers who target caught bluffers.
                            Some(Box::new(ReaderAgent::random(&mut self.rng)))
                        } else {
                            let cs = 0.4 + 2.1 * self.rng.f64() * self.rng.f64();
                            Some(Box::new(HumanAgent::random_traits(cs, &mut self.rng)))
                        }
                    } else if r < h + c && self.champ.is_some() {
                        Some(make_agent("policy", self.champ.as_ref()))
                    } else if r < h + c + p {
                        // Anything from very passive to very aggressive.
                        Some(Box::new(ProbAgent {
                            base_threshold: 0.3 + 0.55 * self.rng.f64(),
                            risk_weight: 0.7 * self.rng.f64(),
                            bluff_rate: 0.5 * self.rng.f32() * self.rng.f32(),
                        }))
                    } else if r < h + c + p + hon {
                        Some(make_agent("honest", None))
                    } else {
                        snap[q] = r < h + c + p + hon + sn;
                        None
                    };
                    net[q] = bots[q].is_none();
                }
                // At least one learning seat per game (seat 0 is alive in short-handed games).
                if !(0..n).any(|q| net[q] && !snap[q] && g.alive[q]) {
                    net[0] = true;
                    snap[0] = false;
                    bots[0] = None;
                }
                if short < 4 {
                    for q in 0..n {
                        if !g.alive[q] {
                            net[q] = false;
                            snap[q] = false;
                            bots[q] = Some(make_agent("honest", None));
                        }
                    }
                    // Mostly with a fast dumper among the other survivors.
                    let others: Vec<usize> = (1..n).filter(|&q| g.alive[q]).collect();
                    // Not in ranked lobbies: ranked 1v1 players bluff little (called plays lies 29%), and
                    // forced dumpers there taught the network to over-call 1v1 single cards.
                    if !g.ranked && self.rng.f32() < 0.7 {
                        let d = others[self.rng.below(others.len() as u32) as usize];
                        net[d] = false;
                        snap[d] = false;
                        bots[d] = Some(Box::new(DumperAgent::random(&mut self.rng)));
                    }
                    if !(0..n).any(|q| net[q] && !snap[q]) {
                        net[0] = true;
                        snap[0] = false;
                        bots[0] = None;
                    }
                }
            }
        }
        for q in 0..n {
            g.net_seat[q] = net[q];
        }
        self.warm_up(&mut g, &mut bots);
        let mut s = Slot { g, net, snap, bots, stage: 0, pending: None, base: [0.0; MAX_PLAYERS], round_ids: Vec::new() };
        self.round_start(&mut s);
        s
    }

    /// Earlier games with the same players: 0-30 hidden rounds played by the same bots (the
    /// network's seats by a human-like stand-in), whose public record seeds this game's. Each
    /// seat is "met before" half the time and a stranger otherwise; in the advisor about 44% of
    /// opponents are repeats.
    fn warm_up(&mut self, g: &mut Game, bots: &mut [Option<Box<dyn Agent>>]) {
        let rounds = self.rng.below(31);
        if rounds == 0 {
            return;
        }
        let n = g.n;
        let mut stand: Vec<Box<dyn Agent>> = (0..n)
            .map(|_| Box::new(HumanAgent { ranked: g.ranked, ..HumanAgent::random_traits(0.4 + 2.1 * self.rng.f64() * self.rng.f64(), &mut self.rng) }) as Box<dyn Agent>)
            .collect();
        let mut prof = [engine::Prof::default(); MAX_PLAYERS];
        let mut rec = [[0u16; 5]; MAX_PLAYERS];
        let mut w = Game::new_game(n, self.devil, &mut self.rng);
        w.ranked = g.ranked;
        let add = |w: &Game, prof: &mut [engine::Prof; MAX_PLAYERS], rec: &mut [[u16; 5]; MAX_PLAYERS]| {
            for q in 0..n {
                prof[q].add(&w.prof[q]);
                for (i, v) in [w.rec_called[q], w.rec_lies[q], w.rec_opps[q], w.rec_calls[q], w.rec_calls_right[q]].iter().enumerate() {
                    rec[q][i] += *v as u16;
                }
            }
        };
        let mut done = 0;
        let mut steps = 0;
        while done < rounds && steps < 5000 {
            steps += 1;
            let p = w.current;
            let rng = &mut self.rng;
            let bot: &mut Box<dyn Agent> = match bots[p].as_mut() {
                Some(b) => b,
                None => &mut stand[p],
            };
            if w.deal_legal() && bot.decide_deal(&w, p, rng) {
                w.apply_deal(rng);
            } else if w.is_forced(p) {
                w.apply_call();
            } else if w.last.is_some() && bot.decide_call(&w, p, rng) {
                w.apply_call();
            } else {
                let a = bot.decide_play(&w, p, rng);
                w.apply_action(a);
            }
            if w.round_over {
                w.finish_round(RULE, rng);
                done += 1;
                if w.game_over {
                    add(&w, &mut prof, &mut rec);
                    w = Game::new_game(n, self.devil, &mut self.rng);
                }
            }
        }
        add(&w, &mut prof, &mut rec);
        for q in 0..n {
            if self.rng.f32() < 0.5 {
                continue;
            }
            g.prof[q].add(&prof[q]);
            let c = |v: u16, cur: u8| (v + cur as u16).min(255) as u8;
            g.rec_called[q] = c(rec[q][0], g.rec_called[q]);
            g.rec_lies[q] = c(rec[q][1], g.rec_lies[q]);
            g.rec_opps[q] = c(rec[q][2], g.rec_opps[q]);
            g.rec_calls[q] = c(rec[q][3], g.rec_calls[q]);
            g.rec_calls_right[q] = c(rec[q][4], g.rec_calls_right[q]);
        }
    }

    fn round_start(&self, s: &mut Slot) {
        for q in 0..s.g.n {
            if s.net[q] && s.g.alive[q] {
                s.base[q] = self.value.state_value(&s.g.alive, &s.g.shots, s.g.starter, q);
            }
        }
        s.round_ids.clear();
    }

    /// Pays every network decision of the finished round its seat's change in win probability,
    /// then deals the next round (or records the game result).
    fn finish_round(&mut self, s: &mut Slot) {
        for &(q, id) in &s.round_ids {
            let v = self.value.after_round(&s.g, q, RULE) - s.base[q];
            self.rewards.push((id, v as f32));
        }
        s.g.finish_round(RULE, &mut self.rng);
        s.stage = 0;
        if s.g.game_over {
            self.games += 1;
            if s.net[s.g.winner] {
                self.net_wins += 1.0;
            }
        } else {
            self.round_start(s);
        }
    }

    /// Plays bots until this game waits on a network decision.
    fn advance(&mut self, s: &mut Slot) {
        let mut steps = 0;
        while s.pending.is_none() {
            steps += 1;
            if s.g.game_over || steps > 10_000 {
                *s = self.new_slot();
                steps = 0;
                continue;
            }
            let p = s.g.current;
            if s.net[p] {
                if s.stage == 0 {
                    if s.g.deal_legal() {
                        s.pending = Some((p, KIND_DEAL, self.next_id));
                        self.next_id += 1;
                        return;
                    }
                    s.stage = 1;
                }
                if s.stage == 1 {
                    if s.g.is_forced(p) {
                        s.g.apply_call();
                        if s.g.round_over {
                            self.finish_round(s);
                        }
                        continue;
                    }
                    if s.g.last.is_some() {
                        s.pending = Some((p, KIND_CALL, self.next_id));
                        self.next_id += 1;
                        return;
                    }
                    s.stage = 2;
                }
                s.pending = Some((p, KIND_PLAY, self.next_id));
                self.next_id += 1;
                return;
            }
            let rng = &mut self.rng;
            let bot = s.bots[p].as_mut().expect("bot seat");
            if s.g.deal_legal() && bot.decide_deal(&s.g, p, rng) {
                s.g.apply_deal(rng);
            } else if s.g.is_forced(p) {
                s.g.apply_call();
            } else if s.g.last.is_some() && bot.decide_call(&s.g, p, rng) {
                s.g.apply_call();
            } else {
                let a = bot.decide_play(&s.g, p, rng);
                s.g.apply_action(a);
            }
            if s.g.round_over {
                self.finish_round(s);
            }
        }
    }

    fn apply(&mut self, s: &mut Slot, a: usize) {
        let Some((p, kind, id)) = s.pending.take() else { return };
        s.round_ids.push((p, id));
        match kind {
            KIND_DEAL => {
                if a == 1 {
                    s.g.apply_deal(&mut self.rng);
                    s.stage = 0;
                } else {
                    s.stage = 1;
                }
            }
            KIND_CALL => {
                if a == 1 {
                    s.g.apply_call();
                    s.stage = 0;
                } else {
                    s.stage = 2;
                }
            }
            _ => {
                let lg = &LEGAL[s.g.hm[p] as usize][s.g.hn[p] as usize];
                let n = s.g.n_actions(p);
                let a = a.min(n - 1);
                let act = if a < lg.n { undominate(s.g.hn[p], lg.acts[a]) } else { DEVIL_PLAY };
                s.g.apply_action(act);
                s.stage = 0;
            }
        }
        if s.g.round_over {
            self.finish_round(s);
        }
    }

    fn observation(&self, py: Python<'_>) -> PyResult<PyObject> {
        let b = self.slots.len();
        let mut ids = Vec::with_capacity(b);
        let mut feats = Vec::with_capacity(b * N_FEAT);
        let mut kinds = Vec::with_capacity(b);
        let mut snaps = Vec::with_capacity(b);
        let mut hist = Vec::with_capacity(b * HIST_LEN * EV_DIM);
        let mut hlen = Vec::with_capacity(b);
        let mut masks = Vec::with_capacity(b * MAX_ACTIONS);
        for s in &self.slots {
            let (p, kind, id) = s.pending.expect("every game waits on the network after advance");
            ids.push(id);
            feats.extend(encode(&s.g, p, kind));
            kinds.push(kind);
            snaps.push(s.snap[p] as u8);
            let (h, n) = encode_hist(&s.g, p);
            hist.extend(h);
            hlen.push(n as u8);
            let legal = if kind == KIND_PLAY { s.g.n_actions(p) } else { 2 };
            masks.extend((0..MAX_ACTIONS).map(|k| (k < legal) as u8 as f32));
        }
        let feats = PyArray1::from_vec_bound(py, feats).reshape([b, N_FEAT])?;
        let masks = PyArray1::from_vec_bound(py, masks).reshape([b, MAX_ACTIONS])?;
        let hist = PyArray1::from_vec_bound(py, hist).reshape([b, HIST_LEN, EV_DIM])?;
        Ok((PyArray1::from_vec_bound(py, ids), feats, PyArray1::from_vec_bound(py, kinds), masks, PyArray1::from_vec_bound(py, snaps), hist, PyArray1::from_vec_bound(py, hlen)).into_py(py))
    }
}

#[pymethods]
impl VecEnv {
    /// n_games games side by side. `champion`: table policy file for champion seats ("" none).
    /// mix: shares of human-like, champion and probability-bot seats in training (rest network).
    /// eval_opp: if set, seat 0 is the network and the others are that bot ("champ", "human",
    /// "heavy", "honestbig", "bluffer", "prob", ...).
    #[new]
    #[pyo3(signature = (n_games, devil, champion, seed, mix=(0.35, 0.2, 0.1, 0.1, 0.1), eval_opp=None))]
    fn py_new(n_games: usize, devil: bool, champion: &str, seed: u64, mix: (f32, f32, f32, f32, f32), eval_opp: Option<String>) -> PyResult<Self> {
        let champ = if champion.is_empty() {
            None
        } else {
            Some(Arc::new(Policy::load(champion).map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{champion}: {e}")))?))
        };
        let mut env = VecEnv {
            slots: Vec::new(),
            rng: Rng::new(seed),
            value: ValueModel::new(4),
            champ,
            devil,
            mix,
            eval_opp,
            next_id: 0,
            rewards: Vec::new(),
            games: 0,
            net_wins: 0.0,
        };
        for _ in 0..n_games {
            let mut s = env.new_slot();
            env.advance(&mut s);
            env.slots.push(s);
        }
        Ok(env)
    }

    /// (ids, features [B,104], kinds [B], legal masks [B,10], frozen-past-network flags [B]) for
    /// the decision each game waits on.
    fn observe(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.observation(py)
    }

    /// Applies one action per game (in observe order) and returns the next observation.
    fn step(&mut self, py: Python<'_>, actions: Vec<u32>) -> PyResult<PyObject> {
        let mut slots = std::mem::take(&mut self.slots);
        for (s, &a) in slots.iter_mut().zip(&actions) {
            self.apply(s, a as usize);
            self.advance(s);
        }
        self.slots = slots;
        self.observation(py)
    }

    /// Rewards for decisions whose round has finished since the last call: (ids, rewards).
    fn drain_rewards(&mut self, py: Python<'_>) -> PyObject {
        let (ids, r): (Vec<u64>, Vec<f32>) = self.rewards.drain(..).unzip();
        (PyArray1::from_vec_bound(py, ids), PyArray1::from_vec_bound(py, r)).into_py(py)
    }

    /// (finished games, games won by a network seat), and reset the counters.
    fn take_results(&mut self) -> (u64, f64) {
        let r = (self.games, self.net_wins);
        self.games = 0;
        self.net_wins = 0.0;
        r
    }
}

/// The play options for a hand with `hm` real and `hn` fake cards, in action-index order:
/// (real cards thrown, fake cards thrown). A Devil play, when held, comes after these.
#[pyfunction]
fn legal_plays(hm: u8, hn: u8) -> Vec<(u8, u8)> {
    let lg = &LEGAL[hm as usize][hn as usize];
    lg.acts[..lg.n].to_vec()
}

#[pymodule]
fn pyliars(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    m.add_function(wrap_pyfunction!(legal_plays, m)?)?;
    m.add("N_FEAT", N_FEAT)?;
    Ok(())
}
