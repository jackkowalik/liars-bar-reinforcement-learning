//! Data-fitted call rule (v61, 2026-11-06): P(this claim is a lie) from a logistic model fitted on
//! 3,996 labelled human claims the bot faced (research/fit_call_v61.py -> research/call_model_v61.json).
//! The features are computed exactly as in research/call_scorer.py, so the offline scores carry over.
//! Used by the live advisor (LIARS_CALL_RULE=data) and by the simulator wrapper (`fiteval --datacall`).
use std::collections::HashMap;

use crate::engine::Game;

pub const NF: usize = 18;

pub struct DataCall {
    pub w: Vec<f64>,
    pub b: f64,
    pub mu: Vec<f64>,
    pub sd: Vec<f64>,
    pub p0: f64,
    pub k: f64,
    pub thresh: f64,
    /// Player name -> (labelled plays in the logs, lies among them).
    pub priors: HashMap<String, (f64, f64)>,
}

impl DataCall {
    pub fn load(path: &str) -> Result<DataCall, String> {
        let txt = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let v: serde_json::Value = serde_json::from_str(&txt).map_err(|e| format!("{path}: {e}"))?;
        let arr = |k: &str| -> Result<Vec<f64>, String> {
            v[k].as_array().ok_or(format!("{path}: missing {k}"))?.iter().map(|x| x.as_f64().ok_or(format!("{path}: bad {k}"))).collect()
        };
        let w = arr("w")?;
        let mu = arr("mu")?;
        let sd = arr("sd")?;
        if w.len() != NF || mu.len() != NF || sd.len() != NF {
            return Err(format!("{path}: expected {NF} features, got {}", w.len()));
        }
        let mut priors = HashMap::new();
        if let Some(obj) = v["priors"].as_object() {
            for (name, nk) in obj {
                if let (Some(n), Some(k)) = (nk[0].as_f64(), nk[1].as_f64()) {
                    priors.insert(name.clone(), (n, k));
                }
            }
        }
        Ok(DataCall {
            w,
            b: v["b"].as_f64().unwrap_or(0.0),
            mu,
            sd,
            p0: v["p0"].as_f64().unwrap_or(0.43),
            k: v["K"].as_f64().unwrap_or(10.0),
            thresh: v["thresh"].as_f64().unwrap_or(0.6),
            priors,
        })
    }

    /// Shrunk bluff rate for a player name and the number of labelled plays it rests on.
    pub fn prior_info(&self, name: &str) -> (f64, f64) {
        match self.priors.get(name) {
            Some(&(n, k)) => ((k + self.p0 * self.k) / (n + self.k), n),
            None => (self.p0, 0.0),
        }
    }

    pub fn p_lie(&self, f: &[f64; NF]) -> f64 {
        let mut z = self.b;
        for i in 0..NF {
            z += self.w[i] * (f[i] - self.mu[i]) / self.sd[i];
        }
        1.0 / (1.0 + (-z).exp())
    }

    /// Mean think time in the fit (the simulator has no think times).
    pub fn mean_think(&self) -> f64 {
        self.mu[8]
    }
}

fn comb(n: f64, k: f64) -> f64 {
    if k < 0.0 || k > n {
        return 0.0;
    }
    let mut r = 1.0;
    let mut i = 0.0;
    while i < k {
        r *= (n - i) / (i + 1.0);
        i += 1.0;
    }
    r
}

/// Features for the claim `g.last` as seen by `me`, in call_scorer.py's definitions:
/// n, left, left0, plays_before, total_before, claimer_earlier, n_alive, holders, think,
/// my_cards, my_real, my_real0, my_devil, impossible, p_true, prior, rec_called, rec_lies.
/// `rec_*` are this game's public record of the claimer (times called, lies among them).
pub fn features(g: &Game, me: usize, think: f64, prior: f64, rec_called: u8, rec_lies: u8) -> Option<[f64; NF]> {
    let lp = g.last?;
    let c = lp.player;
    if c == me || g.hand(me) == 0 {
        return None;
    }
    let n = lp.n as f64;
    let left = g.hand(c) as f64;
    let plays_before = (g.round_plays.saturating_sub(1) as f64).min(8.0);
    let total_before_raw = g.claims_total.saturating_sub(lp.n) as f64;
    let total_before = total_before_raw.min(12.0);
    // Earlier claims by the claimer this round, from the public event history (kind 1 claims since
    // the last round start, kind 5), skipping the current claim.
    let mut claimer_earlier = 0.0;
    let mut skipped_current = false;
    for ev in g.hist.iter().rev() {
        if ev.kind == 5 {
            break;
        }
        if ev.kind == 1 {
            if !skipped_current {
                skipped_current = true;
                continue;
            }
            if ev.actor as usize == c {
                claimer_earlier += ev.n as f64;
            }
        }
    }
    let n_alive = g.n_alive() as f64;
    // The scorer took the table state before the throw: the claimer always counts as a holder.
    let holders = (0..g.n).filter(|&q| g.alive[q] && (g.hand(q) > 0 || q == c)).count() as f64;
    let my_cards = g.hand(me) as f64;
    let my_real = g.hm[me] as f64;
    let my_devil = g.hd[me] as f64;
    let truths = if n_alive >= 3.0 { 7.0 } else { 8.0 };
    let impossible = if total_before_raw + n + my_real > truths { 1.0 } else { 0.0 };
    // Cards I cannot see (scorer definition: others' cards before the throw, plus the throw).
    let others: f64 = (0..g.n).filter(|&q| q != me && g.alive[q]).map(|q| g.hand(q) as f64 + if q == c { n } else { 0.0 }).sum::<f64>() + n;
    let unseen = others.max(1.0);
    let reals_unseen = (truths - my_real - total_before_raw).max(0.0);
    let p_true = if n > reals_unseen { 0.0 } else { comb(reals_unseen, n) / comb(unseen, n).max(1.0) };
    Some([
        n,
        left,
        if left == 0.0 { 1.0 } else { 0.0 },
        plays_before,
        total_before,
        claimer_earlier,
        n_alive,
        holders,
        think.min(30.0),
        my_cards,
        my_real,
        if my_real == 0.0 { 1.0 } else { 0.0 },
        my_devil,
        impossible,
        p_true,
        prior,
        (rec_called as f64).min(8.0),
        (rec_lies as f64).min(8.0),
    ])
}
