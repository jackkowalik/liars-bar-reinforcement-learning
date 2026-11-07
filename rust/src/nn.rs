//! Neural-network groundwork: a fixed-size numeric encoding of a decision, and a data generator
//! that records the current champion policy's choices for imitation training on the GPU.
//!
//! Unlike the table keys, nothing here is bucketed into a few levels: the network sees exact
//! counts, every player's state in seat order, records and card-count odds, and learns which
//! of them matter.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::Arc;

use crate::agents::{hypergeom_at_least, Agent, HumanAgent, ProbAgent};
use crate::engine::{Game, StarterRule, DECK_SIZE, HAND_SIZE, MAX_ACTIONS};
use crate::policy::{call_key, deal_fields, deal_key, play_key, slack, Policy, DEAL_DIMS};
use crate::rng::Rng;

/// Numbers per encoded decision.
pub const N_FEAT: usize = 135;

/// Decision kinds (also the head the network answers with).
pub const KIND_DEAL: u8 = 0;
pub const KIND_CALL: u8 = 1;
pub const KIND_PLAY: u8 = 2;

fn one_hot(out: &mut Vec<f32>, i: usize, n: usize) {
    for k in 0..n {
        out.push((k == i) as u8 as f32);
    }
}

/// Smoothed rate (k+1)/(n+2), and the sample size squashed to 0..1.
fn rate(out: &mut Vec<f32>, k: u8, n: u8) {
    out.push((k as f32 + 1.0) / (n as f32 + 2.0));
    out.push(n as f32 / (n as f32 + 5.0));
}

/// rate() for the wider habit counters.
fn rate16(out: &mut Vec<f32>, k: u16, n: u16) {
    out.push((k as f32 + 1.0) / (n as f32 + 2.0));
    out.push(n as f32 / (n as f32 + 5.0));
}

/// Encodes the state seen by `p` (their own cards plus public information) as N_FEAT numbers.
pub fn encode(g: &Game, p: usize, kind: u8) -> Vec<f32> {
    let mut f = Vec::with_capacity(N_FEAT);
    let pop = (DECK_SIZE - HAND_SIZE) as u64;
    let hs = HAND_SIZE as u64;
    // Decision and table.
    one_hot(&mut f, kind as usize, 3);
    one_hot(&mut f, g.n_alive().clamp(2, 4) - 2, 3);
    f.push(g.devil_mode as u8 as f32);
    f.push(g.devil_active as u8 as f32);
    f.push(g.claims_total as f32 / 15.0);
    f.push(g.is_forced(p) as u8 as f32);
    // The last claim, its claimer's seat relative to me, and the card-count odds it's true.
    let unseen = g.truths().saturating_sub(g.hm[p] + g.played_match[p]);
    f.push(unseen as f32 / 8.0);
    f.push((slack(g, p).clamp(-3, 6) as f32) / 6.0);
    match g.last {
        Some(lp) => {
            f.push(1.0);
            one_hot(&mut f, lp.n.clamp(1, 3) as usize - 1, 3);
            let rel = (lp.player + g.n - p) % g.n;
            one_hot(&mut f, rel.clamp(1, 3) - 1, 3);
            let all = hypergeom_at_least(g.claims[lp.player] as u64, pop, unseen as u64, hs);
            let this = hypergeom_at_least(lp.n as u64, pop, unseen as u64, hs).max(all);
            f.push(all as f32);
            f.push(this as f32);
        }
        None => f.extend([0.0; 9]),
    }
    // Devil's Deal state.
    f.push(g.deal_legal() as u8 as f32);
    if g.deal_legal() {
        let d = deal_fields(g, p);
        for (v, dim) in d.iter().zip(DEAL_DIMS) {
            f.push(*v as f32 / (dim - 1).max(1) as f32);
        }
    } else {
        f.extend([0.0; 9]);
    }
    // Me.
    f.push(g.hm[p] as f32 / 5.0);
    f.push(g.hn[p] as f32 / 5.0);
    f.push(g.hd[p] as f32);
    one_hot(&mut f, g.shots[p].min(5) as usize, 6);
    f.push(g.claims[p] as f32 / 5.0);
    f.push(g.played_match[p] as f32 / 5.0);
    f.push(g.played_devil[p] as u8 as f32);
    rate(&mut f, g.rec_lies[p], g.rec_called[p]);
    rate(&mut f, g.rec_calls[p], g.rec_opps[p]);
    // Every other seat in turn order from me (next, second, third); missing seats are zeros.
    for r in 1..4 {
        if r >= g.n {
            f.extend([0.0; 19]);
            continue;
        }
        let q = (p + r) % g.n;
        let alive = g.alive[q];
        f.push(alive as u8 as f32);
        if !alive {
            f.extend([0.0; 18]);
            continue;
        }
        let h = g.hand(q);
        f.push(h as f32 / 5.0);
        f.push((h == 0) as u8 as f32);
        f.push((h > 0 && h <= 3) as u8 as f32);
        one_hot(&mut f, g.shots[q].min(5) as usize, 6);
        f.push(g.claims[q] as f32 / 5.0);
        f.push(g.last.is_some_and(|lp| lp.player == q) as u8 as f32);
        rate(&mut f, g.rec_lies[q], g.rec_called[q]);
        rate(&mut f, g.rec_calls[q], g.rec_opps[q]);
        rate(&mut f, g.rec_calls_right[q], g.rec_calls[q]);
        f.push(g.played_devil[q] as u8 as f32);
    }
    // Their habits from this game and earlier ones (appended last, so networks trained before
    // them read only the first 104 numbers): opening calls, fast dumps, lies on hand-emptying
    // claims, lies in 1v1, 3-card claims.
    for r in 1..4 {
        if r >= g.n || !g.alive[(p + r) % g.n] {
            f.extend([0.0; 10]);
            continue;
        }
        let pr = &g.prof[(p + r) % g.n];
        rate16(&mut f, pr.open_calls, pr.open_opps);
        rate16(&mut f, pr.dumps, pr.rounds);
        rate16(&mut f, pr.empty_lies, pr.empty_called);
        rate16(&mut f, pr.duel_lies, pr.duel_called);
        rate16(&mut f, pr.claims3, pr.claims);
    }
    // Ranked lobby (appended last; older networks read only their first inputs).
    f.push(g.ranked as u8 as f32);
    assert_eq!(f.len(), N_FEAT, "encoder size");
    f
}

/// One recorded decision: kind, features, the champion's probabilities and the legal mask.
fn write_sample(w: &mut impl Write, kind: u8, feat: &[f32], probs: &[f32], legal: usize) -> std::io::Result<()> {
    let mut row = Vec::with_capacity(1 + N_FEAT + 2 * MAX_ACTIONS);
    row.push(kind as f32);
    row.extend_from_slice(feat);
    for k in 0..MAX_ACTIONS {
        row.push(if k < probs.len() { probs[k] } else { 0.0 });
    }
    for k in 0..MAX_ACTIONS {
        row.push((k < legal) as u8 as f32);
    }
    for v in row {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Floats per sample row in the data file.
pub const ROW: usize = 1 + N_FEAT + 2 * MAX_ACTIONS;

/// Plays games with the champion in about half the seats and log-fitted human-like players in
/// the rest, recording every champion decision it has a table entry for. Returns samples written.
pub fn generate(policy: Arc<Policy>, devil: bool, samples: usize, seed: u64, out: &str) -> std::io::Result<usize> {
    let mut w = BufWriter::new(File::create(out)?);
    let mut rng = Rng::new(seed);
    let feat = policy.feat();
    let mut written = 0usize;
    while written < samples {
        let n = 4;
        let mut g = Game::new_game(n, devil, &mut rng);
        // Seat kinds: 0 champion, 1 human-like, 2 probability bot.
        let kinds: Vec<u8> = (0..n).map(|_| if rng.f32() < 0.5 { 0 } else if rng.f32() < 0.85 { 1 } else { 2 }).collect();
        let mut humans: Vec<HumanAgent> = (0..n).map(|_| HumanAgent::random_traits(0.4 + 2.1 * rng.f64() * rng.f64(), &mut rng)).collect();
        let mut prob = ProbAgent::default();
        let mut champ = crate::agents::CfrAgent::new(policy.clone());
        let mut steps = 0;
        while !g.game_over && steps < 100_000 {
            steps += 1;
            let p = g.current;
            let k = kinds[p];
            let agent: &mut dyn Agent = match k {
                0 => &mut champ,
                1 => &mut humans[p],
                _ => &mut prob,
            };
            if k == 0 {
                // Record the champion's decision(s) at this turn before it acts.
                if g.deal_legal() {
                    if let Some(pr) = policy.call_probs(deal_key(&g, p)) {
                        write_sample(&mut w, KIND_DEAL, &encode(&g, p, KIND_DEAL), pr, 2)?;
                        written += 1;
                    }
                }
                if g.last.is_some() && !g.is_forced(p) {
                    if let Some(pr) = policy.call_probs(call_key(&g, p, &feat)) {
                        write_sample(&mut w, KIND_CALL, &encode(&g, p, KIND_CALL), pr, 2)?;
                        written += 1;
                    }
                }
                if !g.is_forced(p) {
                    if let Some(pr) = policy.play_probs(play_key(&g, p, &feat)) {
                        let na = g.n_actions(p);
                        write_sample(&mut w, KIND_PLAY, &encode(&g, p, KIND_PLAY), &pr[..na], na)?;
                        written += 1;
                    }
                }
            }
            if g.deal_legal() && agent.decide_deal(&g, p, &mut rng) {
                g.apply_deal(&mut rng);
            } else if g.is_forced(p) {
                g.apply_call();
            } else if g.last.is_some() && agent.decide_call(&g, p, &mut rng) {
                g.apply_call();
            } else {
                let a = agent.decide_play(&g, p, &mut rng);
                g.apply_action(a);
            }
            if g.round_over {
                g.finish_round(StarterRule::Loser, &mut rng);
            }
        }
    }
    w.flush()?;
    Ok(written)
}

/// The trained network, run on the CPU: dense layers with ReLU between them, 14 outputs
/// (Devil's Deal 2, call 2, play 10).
pub struct Net {
    layers: Vec<(usize, usize, Vec<f32>, Vec<f32>)>,
}

impl Net {
    /// Loads `<base>.bin` (weights) using the layer sizes in `<base>.json`.
    pub fn load(base: &str) -> std::io::Result<Net> {
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{base}.json"))?)?;
        let raw = std::fs::read(format!("{base}.bin"))?;
        let floats: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let mut at = 0;
        let mut layers = Vec::new();
        for l in meta["layers"].as_array().ok_or_else(|| std::io::Error::other("bad meta"))? {
            let (i, o) = (l[0].as_u64().unwrap() as usize, l[1].as_u64().unwrap() as usize);
            let w = floats[at..at + i * o].to_vec();
            at += i * o;
            let b = floats[at..at + o].to_vec();
            at += o;
            layers.push((i, o, w, b));
        }
        if at != floats.len() {
            return Err(std::io::Error::other("weight file size does not match the layer sizes"));
        }
        Ok(Net { layers })
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut h = x.to_vec();
        let last = self.layers.len() - 1;
        for (li, (i, o, w, b)) in self.layers.iter().enumerate() {
            let mut out = b.clone();
            for r in 0..*o {
                let row = &w[r * i..(r + 1) * i];
                let mut s = 0.0f32;
                for k in 0..*i {
                    s += row[k] * h[k];
                }
                out[r] += s;
            }
            if li != last {
                out.iter_mut().for_each(|v| *v = v.max(0.0));
            }
            h = out;
        }
        h
    }

    /// Softmax over `logits[a..a+n]`.
    pub fn probs(logits: &[f32], a: usize, n: usize) -> Vec<f32> {
        let m = logits[a..a + n].iter().cloned().fold(f32::MIN, f32::max);
        let e: Vec<f32> = logits[a..a + n].iter().map(|v| (v - m).exp()).collect();
        let s: f32 = e.iter().sum();
        e.into_iter().map(|v| v / s).collect()
    }
}

/// The network to use for "nn" agents (set once by the command that loads it).
pub static NET: std::sync::OnceLock<Arc<Net>> = std::sync::OnceLock::new();

/// Plays with the network.
pub struct NnAgent {
    pub net: Arc<Net>,
}

impl Agent for NnAgent {
    fn decide_deal(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        let out = self.net.forward(&encode(g, p, KIND_DEAL));
        rng.sample(&Net::probs(&out, 0, 2), 2) == 1
    }

    fn decide_call(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        let out = self.net.forward(&encode(g, p, KIND_CALL));
        rng.sample(&Net::probs(&out, 2, 2), 2) == 1
    }

    fn decide_play(&mut self, g: &Game, p: usize, rng: &mut Rng) -> (u8, u8) {
        let na = g.n_actions(p);
        let out = self.net.forward(&encode(g, p, KIND_PLAY));
        let a = rng.sample(&Net::probs(&out, 4, na), na);
        let lg = &crate::engine::LEGAL[g.hm[p] as usize][g.hn[p] as usize];
        if a < lg.n { crate::engine::undominate(g.hn[p], lg.acts[a]) } else { crate::engine::DEVIL_PLAY }
    }
}

/// Events in the transformer's history window, and numbers per event.
pub const HIST_LEN: usize = 48;
pub const EV_DIM: usize = 26;

/// The last HIST_LEN public events seen from `p`'s seat, oldest first and left-padded with zeros,
/// plus how many are real. Per event: kind (claim, call, deal, pull, round start), actor and
/// target seat relative to me (0 = me), claim size or players alive, cards left or pulls taken,
/// round claim total, outcome flag (truth/lie/Devil, deal won, died) and age.
pub fn encode_hist(g: &Game, p: usize) -> (Vec<f32>, usize) {
    let evs = &g.hist[g.hist.len().saturating_sub(HIST_LEN)..];
    let mut out = vec![0.0f32; HIST_LEN * EV_DIM];
    let pad = HIST_LEN - evs.len();
    for (i, e) in evs.iter().enumerate() {
        let row = &mut out[(pad + i) * EV_DIM..(pad + i + 1) * EV_DIM];
        if (1..=5).contains(&e.kind) {
            row[e.kind as usize - 1] = 1.0;
        }
        let rel = |s: u8| (s as usize + g.n - p) % g.n;
        row[5 + rel(e.actor).min(3)] = 1.0;
        if e.kind == 2 || e.kind == 3 {
            row[9 + rel(e.target).min(3)] = 1.0;
        }
        row[13] = if e.kind == 5 { e.n as f32 / 4.0 } else { e.n as f32 / 3.0 };
        row[14] = if e.kind == 4 { e.left as f32 / 6.0 } else { e.left as f32 / 5.0 };
        row[15] = e.total as f32 / 15.0;
        if e.kind == 2 || e.kind == 3 || e.kind == 4 {
            row[16 + (e.flag as usize).min(2)] = 1.0;
        }
        row[19] = (evs.len() - 1 - i) as f32 / HIST_LEN as f32;
        if e.kind == 1 {
            row[20] = e.looks.min(3) as f32 / 3.0;
            row[21] = (e.pace as f32 / 32.0).clamp(-2.0, 2.0) / 2.0;
            row[22] = e.habit as f32 / 255.0;
            row[23] = e.habit_n as f32 / 30.0;
        }
        // Devil's Deal reveal of one play: cards shown, share of them fake, whole play shown.
        if e.kind == 6 {
            row[24] = 1.0;
            row[25] = e.flag as f32 / e.n.max(1) as f32;
        }
    }
    (out, evs.len())
}

/// The transformer network on the CPU (same maths as tnet.py: post-norm layers, ReLU, 4 heads).
pub struct TNet {
    p: std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>,
    layers: usize,
}

fn linear(w: &[f32], b: &[f32], x: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    (0..out_dim).map(|r| b[r] + w[r * in_dim..(r + 1) * in_dim].iter().zip(x).map(|(a, c)| a * c).sum::<f32>()).collect()
}

fn layer_norm(x: &mut [f32], g: &[f32], b: &[f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    for (i, v) in x.iter_mut().enumerate() {
        *v = (*v - mean) * inv * g[i] + b[i];
    }
}

impl TNet {
    pub fn load(base: &str) -> std::io::Result<TNet> {
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(format!("{base}.json"))?)?;
        if meta["kind"].as_str() != Some("tnet") {
            return Err(std::io::Error::other("not a transformer network"));
        }
        let raw = std::fs::read(format!("{base}.bin"))?;
        let floats: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let mut at = 0;
        let mut p = std::collections::HashMap::new();
        for e in meta["params"].as_array().ok_or_else(|| std::io::Error::other("bad meta"))? {
            let name = e[0].as_str().unwrap().to_string();
            let shape: Vec<usize> = e[1].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            let n: usize = shape.iter().product::<usize>().max(1);
            p.insert(name, (shape, floats[at..at + n].to_vec()));
            at += n;
        }
        if at != floats.len() {
            return Err(std::io::Error::other("weight file size does not match the shapes"));
        }
        let layers = (0..).take_while(|l| p.contains_key(&format!("enc.layers.{l}.norm1.weight"))).count();
        Ok(TNet { p, layers })
    }

    fn w(&self, n: &str) -> &[f32] {
        &self.p[n].1
    }

    /// 14 policy logits (deal 2, call 2, play 10) for the situation `x` and the history.
    pub fn forward(&self, x: &[f32], hist: &[f32], hlen: usize) -> Vec<f32> {
        let relu = |v: Vec<f32>| v.into_iter().map(|a| a.max(0.0)).collect::<Vec<f32>>();
        // Networks trained before the habit features read only the first numbers.
        let x_in = self.w("state.0.weight").len() / 256;
        let s = relu(linear(self.w("state.0.weight"), self.w("state.0.bias"), &x[..x_in], 256));
        let s = relu(linear(self.w("state.2.weight"), self.w("state.2.bias"), &s, 256));
        let d = self.p["cls.bias"].1.len();
        let pos = self.w("pos");
        // Tokens: the summary token, then the real events (padding is masked out, so dropped).
        let pad = HIST_LEN - hlen;
        let mut toks: Vec<Vec<f32>> = Vec::with_capacity(hlen + 1);
        let mut t0 = linear(self.w("cls.weight"), self.w("cls.bias"), &s, d);
        t0.iter_mut().enumerate().for_each(|(i, v)| *v += pos[i]);
        toks.push(t0);
        // Networks trained before the tells were added read only the first event features.
        let ev_in = self.w("ev.weight").len() / d;
        for i in pad..HIST_LEN {
            let mut t = linear(self.w("ev.weight"), self.w("ev.bias"), &hist[i * EV_DIM..i * EV_DIM + ev_in], d);
            t.iter_mut().enumerate().for_each(|(j, v)| *v += pos[(i + 1) * d + j]);
            toks.push(t);
        }
        let heads = 4;
        let hd = d / heads;
        for l in 0..self.layers {
            let pre = format!("enc.layers.{l}.");
            let inw = self.w(&format!("{pre}self_attn.in_proj_weight"));
            let inb = self.w(&format!("{pre}self_attn.in_proj_bias"));
            let qkv: Vec<Vec<f32>> = toks.iter().map(|t| linear(inw, inb, t, 3 * d)).collect();
            let n = toks.len();
            let mut att_out = vec![vec![0.0f32; d]; n];
            for h in 0..heads {
                for qi in 0..n {
                    let q = &qkv[qi][h * hd..(h + 1) * hd];
                    let scores: Vec<f32> = (0..n)
                        .map(|ki| q.iter().zip(&qkv[ki][d + h * hd..d + (h + 1) * hd]).map(|(a, b)| a * b).sum::<f32>() / (hd as f32).sqrt())
                        .collect();
                    let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                    let e: Vec<f32> = scores.iter().map(|v| (v - mx).exp()).collect();
                    let z: f32 = e.iter().sum();
                    for ki in 0..n {
                        let wgt = e[ki] / z;
                        let v = &qkv[ki][2 * d + h * hd..2 * d + (h + 1) * hd];
                        for j in 0..hd {
                            att_out[qi][h * hd + j] += wgt * v[j];
                        }
                    }
                }
            }
            let ow = self.w(&format!("{pre}self_attn.out_proj.weight"));
            let ob = self.w(&format!("{pre}self_attn.out_proj.bias"));
            let ff = self.p[&format!("{pre}linear1.bias")].1.len();
            for i in 0..n {
                let sa = linear(ow, ob, &att_out[i], d);
                let mut x1: Vec<f32> = toks[i].iter().zip(&sa).map(|(a, b)| a + b).collect();
                layer_norm(&mut x1, self.w(&format!("{pre}norm1.weight")), self.w(&format!("{pre}norm1.bias")));
                let hmid = relu(linear(self.w(&format!("{pre}linear1.weight")), self.w(&format!("{pre}linear1.bias")), &x1, ff));
                let f2 = linear(self.w(&format!("{pre}linear2.weight")), self.w(&format!("{pre}linear2.bias")), &hmid, d);
                let mut x2: Vec<f32> = x1.iter().zip(&f2).map(|(a, b)| a + b).collect();
                layer_norm(&mut x2, self.w(&format!("{pre}norm2.weight")), self.w(&format!("{pre}norm2.bias")));
                toks[i] = x2;
            }
        }
        let mut cat = s.clone();
        cat.extend_from_slice(&toks[0]);
        let z = relu(linear(self.w("head.0.weight"), self.w("head.0.bias"), &cat, 256));
        linear(self.w("out.weight"), self.w("out.bias"), &z, 14)
    }
}

/// Either network kind, as the advisor uses them.
pub enum Model {
    Mlp(Net),
    T(TNet),
}

impl Model {
    /// Loads a transformer if `<base>.json` says so, otherwise the plain network.
    pub fn load(base: &str) -> std::io::Result<Model> {
        match TNet::load(base) {
            Ok(t) => Ok(Model::T(t)),
            Err(_) => Net::load(base).map(Model::Mlp),
        }
    }

    pub fn is_transformer(&self) -> bool {
        matches!(self, Model::T(_))
    }

    /// The 14 logits for player `p` making a decision of `kind` in `g` (history from g.hist).
    pub fn logits(&self, g: &Game, p: usize, kind: u8) -> Vec<f32> {
        let x = encode(g, p, kind);
        match self {
            Model::Mlp(n) => n.forward(&x),
            Model::T(t) => {
                let (h, n) = encode_hist(g, p);
                t.forward(&x, &h, n)
            }
        }
    }
}


/// Plays with any saved network (plain or transformer); the game must keep its event history.
pub struct ModelAgent {
    pub m: Arc<Model>,
}

impl Agent for ModelAgent {
    fn decide_deal(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        let out = self.m.logits(g, p, KIND_DEAL);
        rng.sample(&Net::probs(&out, 0, 2), 2) == 1
    }

    fn decide_call(&mut self, g: &Game, p: usize, rng: &mut Rng) -> bool {
        let out = self.m.logits(g, p, KIND_CALL);
        rng.sample(&Net::probs(&out, 2, 2), 2) == 1
    }

    fn decide_play(&mut self, g: &Game, p: usize, rng: &mut Rng) -> (u8, u8) {
        let na = g.n_actions(p);
        let out = self.m.logits(g, p, KIND_PLAY);
        let a = rng.sample(&Net::probs(&out, 4, na), na);
        let lg = &crate::engine::LEGAL[g.hm[p] as usize][g.hn[p] as usize];
        if a < lg.n { crate::engine::undominate(g.hn[p], lg.acts[a]) } else { crate::engine::DEVIL_PLAY }
    }
}

#[cfg(test)]
mod parity {
    /// The Rust transformer matches PyTorch on saved inputs (PARITY=<model base>).
    #[test]
    fn tnet_matches_pytorch() {
        let Ok(base) = std::env::var("PARITY") else { return };
        let t = super::TNet::load(&base).unwrap();
        let raw = std::fs::read("C:/liars/nn/parity.bin").unwrap();
        let f: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let row = 1 + 104 + super::HIST_LEN * super::EV_DIM + 14;
        let mut worst = 0.0f32;
        for r in f.chunks_exact(row) {
            let hl = r[0] as usize;
            let out = t.forward(&r[1..105], &r[105..105 + super::HIST_LEN * super::EV_DIM], hl);
            let exp = &r[row - 14..];
            for k in 0..14 {
                worst = worst.max((out[k] - exp[k]).abs());
            }
            println!("history {hl}: rust {:.4} torch {:.4}", out[0], exp[0]);
        }
        println!("largest difference {worst:e}");
        assert!(worst < 1e-3);
    }
}
