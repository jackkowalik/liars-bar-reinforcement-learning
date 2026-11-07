//! "Why" behind the advice: what the other players probably hold, and what each option is worth.
//!
//! Hands for the other players are sampled from the cards I can't see and weighted by how likely
//! a network player holding them would have made this round's claims (and not called). That
//! gives a lie chance that reads the claims, not just the card count. Each option is then played
//! out to the end of the game from those hands, the network playing every seat, with the same
//! hands and dice for every option so the comparison is fair.

use std::sync::Arc;

use crate::engine::{Game, Play, StarterRule, DECK_SIZE, DEVIL_PLAY, HAND_SIZE, LEGAL, MATCHING_IN_DECK};
use crate::nn::{Model, Net, KIND_CALL, KIND_DEAL, KIND_PLAY};
use crate::rng::Rng;

/// One play this round, in turn-order seats: (player, cards, real cards, Devil). Real cards and
/// the Devil are only known for my own plays.
pub type RoundPlay = (usize, u8, u8, bool);

pub struct Analysis {
    /// Chance the last claim is a lie / the Devil, from the weighted hands.
    pub lie: f64,
    pub devil: f64,
    /// Effective number of hand samples behind the read (low = the claims were surprising).
    pub ess: f64,
    /// Option (None = call LIAR, Some(action)) and my game win chance after it.
    pub options: Vec<(Option<(u8, u8)>, f64)>,
}

fn act_of(g: &Game, p: usize, i: usize) -> (u8, u8) {
    let lg = &LEGAL[g.hm[p] as usize][g.hn[p] as usize];
    if i < lg.n { lg.acts[i] } else { DEVIL_PLAY }
}

/// Share of each read opponent's choices treated as random (LIARS_READ_MIX, default 0.4).
fn read_mix() -> f64 {
    std::env::var("LIARS_READ_MIX").ok().and_then(|v| v.parse().ok()).unwrap_or(0.4)
}

fn size_of(a: (u8, u8)) -> u8 {
    if a == DEVIL_PLAY { 1 } else { a.0 + a.1 }
}

/// Deals the other players' round-start hands from the unseen cards, replays this round's plays
/// and returns the game at my decision with its weight (0 when the hands can't explain the plays).
fn replay(base: &Game, me: usize, plays: &[RoundPlay], my_start: (u8, u8, u8), start_size: &[u8], model: &Model, rng: &mut Rng) -> (Game, f64) {
    let mut g = base.clone();
    let devil = g.devil_active as u8;
    let (sm, sn, sd) = my_start;
    let mut rem_match = (MATCHING_IN_DECK - devil).saturating_sub(sm) as u32;
    let mut rem_devil = devil.saturating_sub(sd) as u32;
    let mut rem_total = (DECK_SIZE - sm - sn - sd) as u32;
    for q in 0..g.n {
        g.claims[q] = 0;
        g.played_match[q] = 0;
        g.played_devil[q] = false;
        if q == me {
            g.hm[q] = sm;
            g.hn[q] = sn;
            g.hd[q] = sd;
            continue;
        }
        g.hm[q] = 0;
        g.hn[q] = 0;
        g.hd[q] = 0;
        for _ in 0..start_size[q] {
            if rem_total == 0 {
                break;
            }
            let r = rng.below(rem_total);
            if r < rem_match {
                g.hm[q] += 1;
                rem_match -= 1;
            } else if r < rem_match + rem_devil {
                g.hd[q] += 1;
                rem_devil -= 1;
            } else {
                g.hn[q] += 1;
            }
            rem_total -= 1;
        }
    }
    g.claims_total = 0;
    g.last = None;
    g.n_recent = 0;
    g.round_over = false;
    let mut w = 1.0f64;
    // My real cards played: what's gone from my start hand, spent on my plays in order (the
    // opponents can't see the split; my final hand is set to the real one below).
    let mut my_true = sm.saturating_sub(base.hm[me]);
    let mut my_devil = sd > base.hd[me];
    for &(p, n, _nt, _dv) in plays {
        g.current = p;
        if p == me {
            if n == 1 && my_devil && g.hd[p] > 0 && (my_true == 0 || g.hn[p] == 0) {
                my_devil = false;
                g.apply_devil();
            } else {
                let t = my_true.min(n).min(g.hm[p]);
                my_true -= t;
                let b = (n - t).min(g.hn[p]);
                if t + b == 0 {
                    return (g, 0.0);
                }
                g.apply_play(t, b);
            }
            continue;
        }
        // Real players are noisier than the network: blend its choices with a uniform one.
        let mix = read_mix();
        if g.last.is_some() && !g.is_forced(p) {
            w *= (1.0 - mix) * Net::probs(&model.logits(&g, p, KIND_CALL), 2, 2)[0] as f64 + mix * 0.5;
        }
        let na = g.n_actions(p);
        if na == 0 {
            return (g, 0.0);
        }
        let pr: Vec<f32> = Net::probs(&model.logits(&g, p, KIND_PLAY), 4, na).iter().map(|&x| (1.0 - mix as f32) * x + mix as f32 / na as f32).collect();
        let mut fit: Vec<f32> = (0..na).map(|i| if size_of(act_of(&g, p, i)) == n { pr[i] } else { 0.0 }).collect();
        let s: f32 = fit.iter().sum();
        if s <= 0.0 {
            return (g, 0.0);
        }
        w *= s as f64;
        fit.iter_mut().for_each(|x| *x /= s);
        let i = rng.sample(&fit, na);
        g.apply_action(act_of(&g, p, i));
    }
    g.current = me;
    g.hm[me] = base.hm[me];
    g.hn[me] = base.hn[me];
    g.hd[me] = base.hd[me];
    (g, w)
}

/// Plays the game to the end with the network in every seat; true if `me` wins.
fn rollout(mut g: Game, me: usize, model: &Model, rule: StarterRule, rng: &mut Rng) -> bool {
    let mut steps = 0;
    loop {
        if g.round_over {
            g.finish_round(rule, rng);
        }
        if g.game_over {
            return g.winner == me;
        }
        if !g.alive[me] {
            return false;
        }
        steps += 1;
        if steps > 2000 {
            return false;
        }
        let p = g.current;
        if g.deal_legal() && Net::probs(&model.logits(&g, p, KIND_DEAL), 0, 2)[1] > rng.f32() {
            g.apply_deal(rng);
        } else if g.is_forced(p) {
            g.apply_call();
        } else if g.last.is_some() && Net::probs(&model.logits(&g, p, KIND_CALL), 2, 2)[1] > rng.f32() {
            g.apply_call();
        } else {
            let na = g.n_actions(p);
            let i = rng.sample(&Net::probs(&model.logits(&g, p, KIND_PLAY), 4, na), na);
            let a = act_of(&g, p, i);
            g.apply_action(a);
        }
    }
}

/// `base` is the current public game from my seat (opponents' hands unknown), `plays` this
/// round's plays so far, `my_start` my round-start (matching, other, Devil) counts.
pub fn analyse(base: &Game, me: usize, plays: &[RoundPlay], my_start: (u8, u8, u8), model: &Arc<Model>, rule: StarterRule,
    samples: usize, rollouts: usize) -> Analysis {
    let mut start_size = [0u8; crate::engine::MAX_PLAYERS];
    for q in 0..base.n {
        let played: u8 = plays.iter().filter(|x| x.0 == q).map(|x| x.1).sum();
        start_size[q] = if base.alive[q] { (base.hand(q) + played).min(HAND_SIZE) } else { 0 };
    }
    // Few threads by default so the advisor never stalls the game (LIARS_WHY_THREADS overrides).
    let threads = std::env::var("LIARS_WHY_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(4usize).max(1);
    // 1. Weighted hands.
    let parts: Vec<Vec<(Game, f64)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads).map(|t| {
            let model = model.clone();
            let start_size = &start_size;
            sc.spawn(move || {
                let mut rng = Rng::new(0x9e37 + t as u64);
                (t..samples).step_by(threads).map(|_| replay(base, me, plays, my_start, start_size, &model, &mut rng)).collect()
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let parts: Vec<(Game, f64)> = parts.into_iter().flatten().filter(|x| x.1 > 0.0).collect();
    let sw: f64 = parts.iter().map(|x| x.1).sum();
    let sw2: f64 = parts.iter().map(|x| x.1 * x.1).sum();
    let (mut lie, mut devil) = (0.0, 0.0);
    if let Some(Play { .. }) = base.last {
        for (g, w) in &parts {
            if let Some(lp) = g.last {
                if lp.devil {
                    devil += w;
                } else if lp.n_true < lp.n {
                    lie += w;
                }
            }
        }
    }
    let ess = if sw2 > 0.0 { sw * sw / sw2 } else { 0.0 };
    if sw <= 0.0 {
        return Analysis { lie: f64::NAN, devil: 0.0, ess: 0.0, options: Vec::new() };
    }
    // 2. Resample hands by weight, then try every option on the same hands and dice.
    let mut rng = Rng::new(77);
    let mut pick = Vec::with_capacity(rollouts);
    let step = sw / rollouts as f64;
    let mut u = rng.f64() * step;
    let mut acc = 0.0;
    let mut j = 0;
    for (i, (_, w)) in parts.iter().enumerate() {
        acc += w;
        while u < acc && j < rollouts {
            pick.push(i);
            u += step;
            j += 1;
        }
    }
    let g0 = &parts[pick[0]].0;
    let mut opts: Vec<Option<(u8, u8)>> = Vec::new();
    if base.last.is_some() {
        opts.push(None);
    }
    if !base.is_forced(me) {
        for i in 0..g0.n_actions(me) {
            let a = crate::engine::undominate(g0.hn[me], act_of(g0, me, i));
            if !opts.contains(&Some(a)) {
                opts.push(Some(a));
            }
        }
    }
    let wins: Vec<Vec<u32>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads).map(|t| {
            let model = model.clone();
            let (parts, pick, opts) = (&parts, &pick, &opts);
            sc.spawn(move || {
                let mut won = vec![0u32; opts.len()];
                for k in (t..pick.len()).step_by(threads) {
                    for (oi, o) in opts.iter().enumerate() {
                        let mut g = parts[pick[k]].0.clone();
                        let mut rng = Rng::new(1_000_003 * k as u64 + 17);
                        match o {
                            None => g.apply_call(),
                            Some(a) => g.apply_action(*a),
                        }
                        won[oi] += rollout(g, me, &model, rule, &mut rng) as u32;
                    }
                }
                won
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let n = pick.len().max(1) as f64;
    let options = opts.iter().enumerate().map(|(oi, o)| (*o, wins.iter().map(|w| w[oi]).sum::<u32>() as f64 / n)).collect();
    Analysis { lie: lie / sw, devil: devil / sw, ess, options }
}
