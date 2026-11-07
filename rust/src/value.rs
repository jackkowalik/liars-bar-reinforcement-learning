use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::sync::Arc;

use crate::agents::{play_game, Agent, CfrAgent};
use crate::engine::{death_prob, next_alive_of, Game, StarterRule, CHAMBERS, MAX_PLAYERS};
use crate::policy::{read_u32, read_u64, Policy};
use crate::rng::Rng;

/// Round-start state from one player's seat: my pulls, then the next three seats clockwise
/// (pulls, or 6 for dead / absent), then the starter's offset from me.
pub const VALUE_KEYS: usize = 6 * 7 * 7 * 7 * 4;
const PRIOR_WEIGHT: f64 = 30.0;
const VALUE_MAGIC: &[u8; 8] = b"LDVAL001";

pub fn value_key(alive: &[bool; MAX_PLAYERS], shots: &[u8; MAX_PLAYERS], n: usize, starter: usize, me: usize) -> usize {
    let mut k = shots[me] as usize;
    for r in 1..MAX_PLAYERS {
        let s = if r < n {
            let q = (me + r) % n;
            if alive[q] { shots[q] as usize } else { 6 }
        } else {
            6
        };
        k = k * 7 + s;
    }
    k * 4 + (starter + n - me) % n
}

fn analytic(me: u8, others: &mut Vec<u8>, memo: &mut HashMap<(u8, Vec<u8>), f64>) -> f64 {
    if others.is_empty() {
        return 1.0;
    }
    others.sort_unstable();
    if let Some(&v) = memo.get(&(me, others.clone())) {
        return v;
    }
    let n = (others.len() + 1) as f64;
    let mut total = 0.0;
    let pd = death_prob(me);
    if pd < 1.0 {
        total += (1.0 - pd) * analytic(me + 1, &mut others.clone(), memo);
    }
    for idx in 0..others.len() {
        let s = others[idx];
        let pd = death_prob(s);
        let mut rest = others.clone();
        rest.remove(idx);
        total += pd * analytic(me, &mut rest.clone(), memo);
        if pd < 1.0 {
            rest.push(s + 1);
            total += (1.0 - pd) * analytic(me, &mut rest, memo);
        }
    }
    let v = total / n;
    memo.insert((me, others.clone()), v);
    v
}

/// Win probability at the start of a round. Prior: every future round's loser is uniform among
/// alive players (the old Python value). With a measured table from self-play, the estimate is
/// the empirical win rate shrunk toward the prior by PRIOR_WEIGHT pseudo-games.
pub struct ValueModel {
    pub n: usize,
    prior: Vec<f64>,
    emp: Option<Vec<(f64, f64)>>,
}

impl ValueModel {
    pub fn new(n: usize) -> ValueModel {
        let mut memo = HashMap::new();
        let mut prior = vec![0.0; VALUE_KEYS / 4];
        for (i, v) in prior.iter_mut().enumerate() {
            let me = (i / 343) as u8;
            let mut rest = i % 343;
            let mut others = Vec::new();
            for _ in 0..3 {
                let s = (rest % 7) as u8;
                rest /= 7;
                if s < CHAMBERS {
                    others.push(s);
                }
            }
            *v = if me < CHAMBERS { analytic(me, &mut others, &mut memo) } else { 0.0 };
        }
        ValueModel { n, prior, emp: None }
    }

    pub fn has_empirical(&self) -> bool {
        self.emp.is_some()
    }

    pub fn state_value(&self, alive: &[bool; MAX_PLAYERS], shots: &[u8; MAX_PLAYERS], starter: usize, me: usize) -> f64 {
        if !alive[me] {
            return 0.0;
        }
        if alive[..self.n].iter().filter(|&&a| a).count() <= 1 {
            return 1.0;
        }
        let k = value_key(alive, shots, self.n, starter, me);
        let prior = self.prior[k / 4];
        match &self.emp {
            Some(e) => {
                let (w, v) = e[k];
                (w + PRIOR_WEIGHT * prior) / (v + PRIOR_WEIGHT)
            }
            None => prior,
        }
    }

    /// Expected value for `me` after `loser` pulls the trigger, averaged over the revolver.
    pub fn after_pull(&self, alive: &[bool; MAX_PLAYERS], shots: &[u8; MAX_PLAYERS], loser: usize, me: usize, rule: StarterRule) -> f64 {
        let pd = death_prob(shots[loser]);
        let mut a_dead = *alive;
        a_dead[loser] = false;
        let v_dead = self.with_rule(&a_dead, shots, loser, true, me, rule);
        if pd >= 1.0 {
            return v_dead;
        }
        let mut s_surv = *shots;
        s_surv[loser] += 1;
        let v_surv = self.with_rule(alive, &s_surv, loser, false, me, rule);
        pd * v_dead + (1.0 - pd) * v_surv
    }

    /// Expected value for `me` once a finished round's pulls resolve. A Devil reveal can make
    /// several players pull at once; every live/die combination is weighed exactly.
    pub fn after_round(&self, g: &Game, me: usize, rule: StarterRule) -> f64 {
        if !g.devil_reveal {
            return self.after_pull(&g.alive, &g.shots, g.loser, me, rule);
        }
        let pullers: Vec<usize> = (0..self.n).filter(|&q| g.pullers[q]).collect();
        let mut total = 0.0;
        for mask in 0u32..(1 << pullers.len()) {
            let mut p = 1.0;
            let mut alive = g.alive;
            let mut shots = g.shots;
            for (j, &q) in pullers.iter().enumerate() {
                let pd = death_prob(g.shots[q]);
                if mask >> j & 1 == 1 {
                    p *= pd;
                    alive[q] = false;
                } else {
                    p *= 1.0 - pd;
                    shots[q] += 1;
                }
            }
            if p > 0.0 {
                total += p * self.state_value(&alive, &shots, g.accused, me);
            }
        }
        total
    }

    fn with_rule(&self, alive: &[bool; MAX_PLAYERS], shots: &[u8; MAX_PLAYERS], loser: usize, died: bool, me: usize, rule: StarterRule) -> f64 {
        match rule {
            StarterRule::Loser => {
                let _ = died;
                let s = next_alive_of(alive, self.n, loser);
                self.state_value(alive, shots, s, me)
            }
            StarterRule::Random => {
                let seats: Vec<usize> = (0..self.n).filter(|&q| alive[q]).collect();
                seats.iter().map(|&s| self.state_value(alive, shots, s, me)).sum::<f64>() / seats.len() as f64
            }
        }
    }
}

const SHOT_WEIGHTS: [f64; 6] = [0.30, 0.25, 0.20, 0.13, 0.08, 0.04];

fn pack_state(alive: &[bool; MAX_PLAYERS], shots: &[u8; MAX_PLAYERS], starter: usize) -> u32 {
    let mut k = starter as u32;
    for i in 0..MAX_PLAYERS {
        k |= (alive[i] as u32) << (2 + i);
        k |= (shots[i] as u32) << (6 + 3 * i);
    }
    k
}

fn unpack_state(k: u32) -> ([bool; MAX_PLAYERS], [u8; MAX_PLAYERS], usize) {
    let mut alive = [false; MAX_PLAYERS];
    let mut shots = [0u8; MAX_PLAYERS];
    for i in 0..MAX_PLAYERS {
        alive[i] = (k >> (2 + i)) & 1 == 1;
        shots[i] = ((k >> (6 + 3 * i)) & 7) as u8;
    }
    (alive, shots, (k & 3) as usize)
}

/// Round-start states for training. Without data, a hand-set prior over alive count and
/// pulls; with self-play data, mostly the states real games actually reach.
pub struct StartSampler {
    n: usize,
    states: Vec<u32>,
    cum: Vec<f64>,
    pub empirical_share: f64,
}

impl StartSampler {
    pub fn prior(n: usize) -> StartSampler {
        StartSampler { n, states: Vec::new(), cum: Vec::new(), empirical_share: 0.0 }
    }

    pub fn sample(&self, rng: &mut Rng) -> ([bool; MAX_PLAYERS], [u8; MAX_PLAYERS], usize) {
        if !self.states.is_empty() && rng.f64() < self.empirical_share {
            let total = *self.cum.last().unwrap();
            let r = rng.f64() * total;
            let i = self.cum.partition_point(|&c| c <= r).min(self.states.len() - 1);
            return unpack_state(self.states[i]);
        }
        self.sample_prior(rng)
    }

    fn sample_prior(&self, rng: &mut Rng) -> ([bool; MAX_PLAYERS], [u8; MAX_PLAYERS], usize) {
        let opts: Vec<(usize, f64)> = [(4, 0.45), (3, 0.33), (2, 0.22)].into_iter().filter(|&(k, _)| k <= self.n).collect();
        let tot: f64 = opts.iter().map(|o| o.1).sum();
        let mut r = rng.f64() * tot;
        let mut n_alive = opts.last().unwrap().0;
        for &(k, w) in &opts {
            if r < w {
                n_alive = k;
                break;
            }
            r -= w;
        }
        let mut seats: Vec<usize> = (0..self.n).collect();
        for i in 0..seats.len() {
            let j = i + rng.below((seats.len() - i) as u32) as usize;
            seats.swap(i, j);
        }
        seats.truncate(n_alive);
        let mut alive = [false; MAX_PLAYERS];
        let mut shots = [0u8; MAX_PLAYERS];
        for &s in &seats {
            alive[s] = true;
            let mut r = rng.f64();
            let mut v = 5;
            for (k, &w) in SHOT_WEIGHTS.iter().enumerate() {
                if r < w {
                    v = k;
                    break;
                }
                r -= w;
            }
            shots[s] = v as u8;
        }
        let starter = seats[rng.below(seats.len() as u32) as usize];
        (alive, shots, starter)
    }
}

/// Self-play statistics: (wins, visits) per value key and a histogram of round-start states.
pub struct SelfPlayStats {
    pub n: usize,
    pub values: Vec<(f64, f64)>,
    pub starts: HashMap<u32, u64>,
    pub games: u64,
}

impl SelfPlayStats {
    pub fn new(n: usize) -> Self {
        SelfPlayStats { n, values: vec![(0.0, 0.0); VALUE_KEYS], starts: HashMap::new(), games: 0 }
    }

    fn merge(&mut self, o: SelfPlayStats) {
        for (a, b) in self.values.iter_mut().zip(o.values) {
            a.0 += b.0;
            a.1 += b.1;
        }
        for (k, c) in o.starts {
            *self.starts.entry(k).or_insert(0) += c;
        }
        self.games += o.games;
    }

    pub fn save(&self, path: &str) -> io::Result<()> {
        let tmp = format!("{path}.tmp");
        {
            let mut w = BufWriter::new(File::create(&tmp)?);
            w.write_all(VALUE_MAGIC)?;
            w.write_all(&(self.n as u64).to_le_bytes())?;
            w.write_all(&self.games.to_le_bytes())?;
            for &(wins, visits) in &self.values {
                w.write_all(&wins.to_le_bytes())?;
                w.write_all(&visits.to_le_bytes())?;
            }
            w.write_all(&(self.starts.len() as u64).to_le_bytes())?;
            for (&k, &c) in &self.starts {
                w.write_all(&k.to_le_bytes())?;
                w.write_all(&c.to_le_bytes())?;
            }
            w.flush()?;
        }
        std::fs::rename(&tmp, path)
    }

    pub fn load(path: &str) -> io::Result<SelfPlayStats> {
        let mut r = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != VALUE_MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a value file"));
        }
        let n = read_u64(&mut r)? as usize;
        let games = read_u64(&mut r)?;
        let mut s = SelfPlayStats::new(n);
        s.games = games;
        for v in s.values.iter_mut() {
            v.0 = f64::from_bits(read_u64(&mut r)?);
            v.1 = f64::from_bits(read_u64(&mut r)?);
        }
        let count = read_u64(&mut r)?;
        for _ in 0..count {
            let k = read_u32(&mut r)?;
            let c = read_u64(&mut r)?;
            s.starts.insert(k, c);
        }
        Ok(s)
    }

    pub fn into_models(self, empirical_share: f64) -> (ValueModel, StartSampler) {
        let mut vm = ValueModel::new(self.n);
        vm.emp = Some(self.values);
        let mut states: Vec<(u32, u64)> = self.starts.into_iter().collect();
        states.sort_unstable();
        let mut cum = Vec::with_capacity(states.len());
        let mut acc = 0.0;
        for &(_, c) in &states {
            acc += c as f64;
            cum.push(acc);
        }
        let sampler = StartSampler { n: self.n, states: states.into_iter().map(|s| s.0).collect(), cum, empirical_share };
        (vm, sampler)
    }
}

/// Plays full self-play games with `policy` in every seat across `threads` threads.
pub fn collect_self_play(policy: &Arc<Policy>, n: usize, games: u64, threads: usize, seed: u64, rule: StarterRule) -> SelfPlayStats {
    let devil = policy.devil;
    let per = games.div_ceil(threads as u64);
    let parts: Vec<SelfPlayStats> = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads)
            .map(|th| {
                let policy = policy.clone();
                s.spawn(move || {
                    let mut rng = Rng::new(seed ^ (0x5151_0000 + th as u64).wrapping_mul(0x9E37_79B9));
                    let mut st = SelfPlayStats::new(n);
                    let mut agents: Vec<Box<dyn Agent>> = (0..n).map(|_| Box::new(CfrAgent::new(policy.clone())) as Box<dyn Agent>).collect();
                    let mut visits: Vec<(usize, usize)> = Vec::with_capacity(256);
                    for _ in 0..per {
                        visits.clear();
                        let winner = play_game(&mut agents, &mut rng, rule, devil, |g| {
                            *st.starts.entry(pack_state(&g.alive, &g.shots, g.starter)).or_insert(0) += 1;
                            for p in 0..n {
                                if g.alive[p] {
                                    visits.push((value_key(&g.alive, &g.shots, n, g.starter, p), p));
                                }
                            }
                        });
                        for &(k, p) in &visits {
                            st.values[k].1 += 1.0;
                            if p == winner {
                                st.values[k].0 += 1.0;
                            }
                        }
                        st.games += 1;
                    }
                    st
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut total = SelfPlayStats::new(n);
    for p in parts {
        total.merge(p);
    }
    total
}
