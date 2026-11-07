mod agents;
mod chaos;
mod chaos_watch;
mod datacall;
mod engine;
mod fit_data;
mod nn;
mod policy;
mod reason;
mod rng;
mod train;
mod ui;
mod value;
mod watch;

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::{Duration, Instant};

use agents::evaluate;
use engine::{StarterRule, LEGAL};
use policy::{play_index, Policy, Tables};
use rng::Rng;
use train::{Bots, Trainer};
use value::{collect_self_play, SelfPlayStats, StartSampler, ValueModel};

const USAGE: &str = "\
liars <command> [options]

  train    --iterations N (50000000)  --threads N  --epsilon F (0.6)  --seed N
           --value value.bin   use measured win rates + real start states (from `value`)
           --bot-share F (0.5)  chance each seat is an honest/prob/random bot for the episode
           --full-game-share F (0.5)  episodes starting from a fresh game vs a sampled state
           --fixed-bots   keep default bot styles instead of re-drawing them every episode
           --chaos        Chaos mode (writes chaos.bin / checkpoint_chaos.bin; eval --chaos too)
           --devil        Devil mode rules (one table-rank card is the Devil while 3+ alive)
           --self-image   play decisions also see my own caught-lying record (tested: no gain)
           --human-bots   add human-like callers (logged ~28% call rate) to the opponent pool
           --coarse-calls / --no-claimer-calls / --no-call-accuracy / --after-next / --threat / --full-table / --fast-reads / --call-table   feature switches
           --no-history   ablation: ignore per-player records (saved in the policy file)
           --empirical-share F (0.8)  --resume  --eval-every N (10000000)  --eval-games N (5000)
           --checkpoint checkpoint.bin  --out policy.bin  --table-log N (21: 2M call / 1M play slots)
  value    --games N (200000)  --threads N  --policy policy.bin  --out value.bin
  eval     --policy policy.bin  --agent policy|prob|honest|random  --games N (20000)
           --vs other.bin   play the policy against 3 copies of another policy
  inspect  --policy policy.bin  --games N (50000)
  bench    --rounds N (2000000)
  watch    --policy policy.bin  --devil-policy devil.bin  --logs dir  --once   live advisor from the LiarsTells log

  common   --players N (4)  --starter loser|random (loser)
";

struct Args {
    map: HashMap<String, String>,
}

impl Args {
    fn parse(raw: &[String]) -> Args {
        let mut map = HashMap::new();
        let mut i = 0;
        while i < raw.len() {
            let k = raw[i].trim_start_matches("--").to_string();
            if i + 1 < raw.len() && !raw[i + 1].starts_with("--") {
                map.insert(k, raw[i + 1].clone());
                i += 2;
            } else {
                map.insert(k, "true".into());
                i += 1;
            }
        }
        Args { map }
    }

    fn get<T: std::str::FromStr>(&self, k: &str, default: T) -> T {
        match self.map.get(k) {
            Some(v) => v.parse().unwrap_or_else(|_| panic!("bad value for --{k}: {v}")),
            None => default,
        }
    }

    fn str(&self, k: &str, default: &str) -> String {
        self.map.get(k).cloned().unwrap_or_else(|| default.to_string())
    }

    fn flag(&self, k: &str) -> bool {
        self.map.contains_key(k)
    }
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).saturating_sub(1).max(1)
}

fn rule_of(a: &Args) -> StarterRule {
    let s = a.str("starter", "loser");
    StarterRule::parse(&s).unwrap_or_else(|| panic!("--starter must be loser or random, got {s}"))
}

fn fmt_n(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn cmd_train(a: &Args) {
    if a.flag("chaos") {
        chaos::train(chaos::TrainOpts {
            n: a.get("players", 4usize),
            iterations: a.get("iterations", 50_000_000u64),
            threads: a.get("threads", default_threads()),
            eval_every: a.get("eval-every", 10_000_000u64),
            eval_games: a.get("eval-games", 5000usize),
            seed: a.get("seed", 0u64),
            checkpoint: a.str("checkpoint", "checkpoint_chaos.bin"),
            out: a.str("out", "chaos.bin"),
            table_log: a.get("table-log", 21u32),
            epsilon: a.get("epsilon", 0.6f32),
            bot_share: a.get("bot-share", 0.5f32),
            resume: a.flag("resume"),
        });
        return;
    }
    let n = a.get("players", 4usize);
    let threads = a.get("threads", default_threads());
    let iterations = a.get("iterations", 50_000_000u64);
    let eval_every = a.get("eval-every", 10_000_000u64);
    let eval_games = a.get("eval-games", 5000usize);
    let seed = a.get("seed", 0u64);
    let ckpt = a.str("checkpoint", "checkpoint.bin");
    let out = a.str("out", "policy.bin");
    let rule = rule_of(a);

    let (value, starts) = match a.map.get("value") {
        Some(path) => {
            let st = SelfPlayStats::load(path).unwrap_or_else(|e| panic!("loading {path}: {e}"));
            println!("value model: {} self-play games from {path}", fmt_n(st.games));
            st.into_models(a.get("empirical-share", 0.8f64))
        }
        None => (ValueModel::new(n), StartSampler::prior(n)),
    };

    print!("allocating tables ... ");
    let tables = Tables::new(a.get("table-log", 21u32));
    println!("{} call + {} play info-set slots", fmt_n(tables.call.capacity() as u64), fmt_n(tables.play.capacity() as u64));
    let mut t0 = 0;
    if a.flag("resume") && Path::new(&ckpt).exists() {
        let (t, np) = tables.load_checkpoint(&ckpt).expect("loading checkpoint");
        assert_eq!(np, n, "checkpoint was trained with {np} players");
        t0 = t;
        println!("resumed from {ckpt} at iteration {}", fmt_n(t));
    }
    let trainer = Trainer {
        tables,
        t: AtomicU64::new(t0),
        n_players: n,
        epsilon: a.get("epsilon", 0.6f32),
        rule,
        value,
        starts,
        bot_share: a.get("bot-share", 0.5f32),
        full_game_share: a.get("full-game-share", 0.5f32),
        history: !a.flag("no-history"),
        vary_bots: !a.flag("fixed-bots"),
        devil: a.flag("devil"),
        feat: policy::Feat {
            history: !a.flag("no-history"),
            self_image: !a.flag("no-history") && a.flag("self-image"),
            coarse_calls: a.flag("coarse-calls"),
            claimer_calls: !a.flag("no-claimer-calls"),
            call_accuracy: !a.flag("no-call-accuracy"),
            after_next: a.flag("after-next"),
            threat: a.flag("threat"),
            full_table: a.flag("full-table"),
            fast_reads: a.flag("fast-reads"),
            call_table: a.flag("call-table"),
        },
    };
    let target = t0 + iterations;
    println!("training to {} rounds on {threads} threads (starter rule {:?}, value {}, bot share {}, history {})",
        fmt_n(target), rule, if trainer.value.has_empirical() { "measured" } else { "prior" }, trainer.bot_share, trainer.history);
    let human_bots = a.flag("human-bots");
    let start = Instant::now();
    let mut chunk = 0u64;
    while trainer.t.load(Relaxed) < target {
        let cur = trainer.t.load(Relaxed);
        let end = ((cur / eval_every) + 1) * eval_every;
        let end = end.min(target);
        chunk += 1;
        std::thread::scope(|s| {
            for th in 0..threads {
                let tr = &trainer;
                s.spawn(move || {
                    let mut rng = Rng::new(seed.wrapping_mul(1_000_003) ^ (chunk << 20) ^ th as u64 ^ cur);
                    let mut recs = Vec::with_capacity(64);
                    let mut bots = Bots::default();
                    bots.with_human = human_bots;
                    while tr.t.load(Relaxed) < end {
                        for _ in 0..256 {
                            tr.iterate(&mut rng, &mut recs, &mut bots);
                        }
                    }
                });
            }
            let mut last_t = cur;
            let mut last = Instant::now();
            while trainer.t.load(Relaxed) < end {
                std::thread::sleep(Duration::from_millis(200));
                if last.elapsed() >= Duration::from_secs(10) {
                    let now_t = trainer.t.load(Relaxed);
                    let rate = (now_t - last_t) as f64 / last.elapsed().as_secs_f64();
                    println!("iter {:>14} | {:>11} rounds/s", fmt_n(now_t), fmt_n(rate as u64));
                    last_t = now_t;
                    last = Instant::now();
                }
            }
        });
        let t = trainer.t.load(Relaxed);
        trainer.tables.save_checkpoint(&ckpt, t, n).expect("saving checkpoint");
        let policy = Arc::new(Policy::from_tables(&trainer.tables, t, &trainer.feat, trainer.devil));
        policy.save(&out).expect("saving policy");
        let (c, p) = policy.trained_counts();
        let mut parts = Vec::new();
        let mut miss_txt = String::new();
        for opp in ["prob", "mixed"] {
            let (wr, miss) = evaluate("policy", Some(&policy), None, opp, eval_games, n, t, rule, trainer.devil);
            parts.push(format!("vs {opp}: {wr:.3}"));
            if let Some(m) = miss {
                miss_txt = format!(" | unseen {:.3}%", m * 100.0);
            }
        }
        println!("eval @ {} | {:.0}s | info sets {} call {} play | baseline {:.3} | {}{}",
            fmt_n(t), start.elapsed().as_secs_f64(), fmt_n(c as u64), fmt_n(p as u64),
            1.0 / n as f64, parts.join(" | "), miss_txt);
    }
    println!("done in {:.0}s, policy saved to {out}", start.elapsed().as_secs_f64());
}

fn load_policy(a: &Args) -> Arc<Policy> {
    let path = a.str("policy", "policy.bin");
    Arc::new(Policy::load(&path).unwrap_or_else(|e| panic!("loading {path}: {e}")))
}

fn cmd_value(a: &Args) {
    let n = a.get("players", 4usize);
    let games = a.get("games", 200_000u64);
    let threads = a.get("threads", default_threads());
    let out = a.str("out", "value.bin");
    let policy = load_policy(a);
    let start = Instant::now();
    let st = collect_self_play(&policy, n, games, threads, a.get("seed", 7u64), rule_of(a));
    println!("{} self-play games in {:.1}s, {} distinct round-start states",
        fmt_n(st.games), start.elapsed().as_secs_f64(), fmt_n(st.starts.len() as u64));
    // Compare measured win rates with the uniform-loser prior on a few common states.
    let prior = ValueModel::new(n);
    println!("\nwin prob at round start, 4 alive, I start | prior vs measured (visits)");
    let alive = [true; 4];
    for (me, others) in [(0u8, [0u8, 0, 0]), (2, [0, 0, 0]), (0, [2, 2, 2]), (4, [1, 1, 1]), (0, [4, 4, 4]), (3, [3, 0, 0])] {
        let shots = [me, others[0], others[1], others[2]];
        let k = value::value_key(&alive, &shots, n, 0, 0);
        let (w, v) = st.values[k];
        let measured = if v > 0.0 { format!("{:.3} ({})", w / v, fmt_n(v as u64)) } else { "-".into() };
        println!("  me {me} pulls, others {others:?}: {:.3} vs {measured}", prior.state_value(&alive, &shots, 0, 0));
    }
    st.save(&out).expect("saving value file");
    println!("\nsaved {out}");
}

fn cmd_eval(a: &Args) {
    if let Some(base) = a.map.get("net") {
        let net = nn::Net::load(base).unwrap_or_else(|e| panic!("loading net {base}: {e}"));
        let _ = nn::NET.set(Arc::new(net));
        println!("hero: neural network {base}");
    }
    if a.flag("chaos") {
        let path = a.str("policy", "chaos.bin");
        let p = Arc::new(chaos::ChaosPolicy::load(&path).unwrap_or_else(|e| panic!("loading {path}: {e}")));
        chaos::eval_all(&p, a.get("players", 4usize), a.get("games", 20_000usize), a.get("seed", 12345u64));
        return;
    }
    let n = a.get("players", 4usize);
    let games = a.get("games", 20_000usize);
    let hero = a.str("agent", "policy");
    let rule = rule_of(a);
    let policy = if hero == "policy" { Some(load_policy(a)) } else { None };
    let opp_policy = a.map.get("vs").map(|p| Arc::new(Policy::load(p).unwrap_or_else(|e| panic!("loading {p}: {e}"))));
    let devil = policy.as_ref().or(opp_policy.as_ref()).map_or(a.flag("devil"), |p| p.devil);
    if let Some(p) = &policy {
        println!("policy trained for {} rounds{}", fmt_n(p.iterations), if p.devil { " (Devil mode)" } else { "" });
    }
    let se = (0.25f64 * 0.75 / games as f64).sqrt();
    println!("hero: {hero} | players {n} | games {games} | baseline {:.3} | std err ~{:.3}", 1.0 / n as f64, se);
    let only: Option<Vec<String>> = a.map.get("opps").map(|v| v.split(',').map(|x| x.to_string()).collect());
    for opp in ["human", "heavy", "honestbig", "bluffer", "prob", "honest", "random", "mixed", "aggro", "timid", "newmix", "rankedfit", "self"] {
        if only.as_ref().is_some_and(|o| !o.iter().any(|x| x == opp)) {
            continue;
        }
        if opp == "self" && policy.is_none() && opp_policy.is_none() {
            continue;
        }
        let (wr, miss) = evaluate(&hero, policy.as_ref(), opp_policy.as_ref(), opp, games, n, a.get("seed", 12345u64), rule, devil);
        let miss_txt = miss.map(|m| format!(" | unseen {:.3}%", m * 100.0)).unwrap_or_default();
        let name = if opp == "self" && opp_policy.is_some() { "--vs" } else { opp };
        println!("vs {name:<7} win rate {wr:.3}{miss_txt}");
    }
}


/// Two networks at one table: each game seats two of each in a random order; win rate of A.
/// --a <base> --b <base> [--games N] [--devil] [--threads T]
fn cmd_h2h(a: &Args) {
    let load = |k: &str| Arc::new(nn::Model::load(&a.str(k, "")).unwrap_or_else(|e| panic!("loading --{k}: {e}")));
    let (ma, mb) = (load("a"), load("b"));
    let games = a.get("games", 4000usize);
    let threads = a.get("threads", 16usize);
    let devil = a.flag("devil");
    let rule = rule_of(a);
    let wins: Vec<(usize, usize)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads).map(|t| {
            let (ma, mb) = (ma.clone(), mb.clone());
            sc.spawn(move || {
                let mut rng = Rng::new(1000 + t as u64);
                let (mut w, mut n) = (0usize, 0usize);
                for _ in (t..games).step_by(threads) {
                    let mut is_a = [true, true, false, false];
                    for i in (1..4usize).rev() { let j = rng.below(i as u32 + 1) as usize; is_a.swap(i, j); }
                    let mut ag: Vec<Box<dyn agents::Agent>> = is_a.iter().map(|&x| Box::new(nn::ModelAgent { m: if x { ma.clone() } else { mb.clone() } }) as Box<dyn agents::Agent>).collect();
                    let mut g = engine::Game::new_game(4, devil, &mut rng);
                    g.record = true;
                    let na = g.n_alive() as u8;
                    g.log_ev(engine::Ev { kind: 5, actor: g.starter as u8, n: na, ..Default::default() });
                    while !g.game_over {
                        let p = g.current;
                        if g.deal_legal() && ag[p].decide_deal(&g, p, &mut rng) {
                            g.apply_deal(&mut rng);
                        } else if g.is_forced(p) {
                            g.apply_call();
                        } else if g.last.is_some() && ag[p].decide_call(&g, p, &mut rng) {
                            g.apply_call();
                        } else {
                            let act = ag[p].decide_play(&g, p, &mut rng);
                            g.apply_action(act);
                        }
                        if g.round_over {
                            g.finish_round(rule, &mut rng);
                        }
                    }
                    w += is_a[g.winner] as usize;
                    n += 1;
                }
                (w, n)
            })
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let (w, n) = wins.iter().fold((0, 0), |s, x| (s.0 + x.0, s.1 + x.1));
    let wr = w as f64 / n as f64;
    println!("A wins {w}/{n} = {wr:.3} (even = 0.500, std err {:.3})", (0.25 / n as f64).sqrt());
}

/// Simulated Devil's Deals for simstats: won, lost on a fake last play, lost otherwise, rounds.
static DEALS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The same behaviour table as nn/real_stats.py, from simulated games: one network seat ("me",
/// left out like the real logs leave me out) and three log-fitted human-like players.
fn cmd_simstats(a: &Args) {
    let me_net = Arc::new(nn::Model::load(&a.str("net", "C:/liars/nn/basic_live")).expect("loading --net"));
    let games = a.get("games", 4000usize);
    let threads = a.get("threads", 8usize);
    let devil = a.flag("devil");
    // (size, left after, alive, seq in round, forced, called, lie, round total claimed, had a real card)
    type Row = (u8, u8, usize, u32, bool, bool, bool, u8, bool);
    let rows: Vec<Row> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads).map(|t| {
            let me_net = me_net.clone();
            sc.spawn(move || {
                let mut rng = Rng::new(5000 + t as u64);
                let mut out: Vec<Row> = Vec::new();
                for _ in (t..games).step_by(threads) {
                    // LIARS_SIM_ALLHUMAN=1: no network seat, so the statistics are the scripted
                    // players' alone (the fit compares them with the logged humans).
                    // Default when every player is ranked (the human fit): the fit compares scripted
                    // players with logged humans, so no network seat.
                    let all_human = std::env::var("LIARS_SIM_ALLHUMAN").map(|v| v == "1")
                        .unwrap_or_else(|_| std::env::var("LIARS_ALL_RANKED").map(|v| v == "1").unwrap_or(false));
                    let me = if all_human { 99 } else { rng.below(4) as usize };
                    let mut ag: Vec<Box<dyn agents::Agent>> = (0..4).map(|q| if q == me {
                        Box::new(nn::ModelAgent { m: me_net.clone() }) as Box<dyn agents::Agent>
                    } else {
                        // Heavy callers are the real top decile (2026-11-01: 14% of logged
                        // opponents call over 30% of chances, 8% over 40%).
                        if std::env::var("LIARS_SIM_FIT").map(|v| v == "1").unwrap_or(false) {
                            // The fitted ranked population (research/fit_population.py).
                            Box::new(agents::FitHuman::random(&mut rng)) as Box<dyn agents::Agent>
                        } else if rng.f32() < 0.10 {
                            Box::new(agents::HumanAgent::heavy_caller(&mut rng)) as Box<dyn agents::Agent>
                        } else {
                            Box::new(agents::HumanAgent::random_traits(0.4 + 2.1 * rng.f64() * rng.f64(), &mut rng)) as Box<dyn agents::Agent>
                        }
                    }).collect();
                    let mut g = engine::Game::new_game(4, devil, &mut rng);
                    g.record = me_net.is_transformer();
                    let mut pend: Option<Row> = None;
                    let mut seq = 0u32;
                    while !g.game_over {
                        let p = g.current;
                        if g.deal_legal() && ag[p].decide_deal(&g, p, &mut rng) {
                            // Deals as the logs can tell them apart: won (several shoot), lost
                            // on a fake last play (caller shoots though the last play lied).
                            let last_fake = g.last.is_some_and(|l| !l.devil && l.n_true < l.n);
                            g.apply_deal(&mut rng);
                            DEALS[if g.lie { 0 } else if last_fake { 1 } else { 2 }].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        } else if g.last.is_some() && (g.is_forced(p) || ag[p].decide_call(&g, p, &mut rng)) {
                            if let Some(mut r) = pend.take() {
                                let lp = g.last.unwrap();
                                r.5 = true;
                                r.6 = !lp.devil && lp.n_true < lp.n; // the logs count a revealed Devil card as not a lie
                                out.push(r);
                            }
                            g.apply_call();
                        } else {
                            if let Some(r) = pend.take() {
                                out.push(r);
                            }
                            let had_real = g.hm[p] > 0;
                            let act = ag[p].decide_play(&g, p, &mut rng);
                            g.apply_action(act);
                            seq += 1;
                            if p != me {
                                let holders = (0..4).filter(|&q| g.alive[q] && g.hand(q) > 0).count();
                                let size = if act == engine::DEVIL_PLAY { 1 } else { act.0 + act.1 };
                                // Truth of the play is known now (the logs only learn it when called).
                                let lie_now = g.last.is_some_and(|l| !l.devil && l.n_true < l.n);
                                pend = Some((size, g.hand(p), g.n_alive(), seq, holders <= 1, false, lie_now, g.claims_total, had_real));
                            }
                        }
                        if g.round_over {
                            pend = None;
                            seq = 0;
                            DEALS[3].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            g.finish_round(StarterRule::Loser, &mut rng);
                        }
                    }
                }
                out
            })
        }).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let unforced: Vec<&Row> = rows.iter().filter(|r| !r.4).collect();
    if devil {
        let d: Vec<u64> = DEALS.iter().map(|x| x.load(std::sync::atomic::Ordering::Relaxed)).collect();
        let r = d[3].max(1) as f64;
        println!("Devil's Deals: {:.1}% of rounds (won {:.2}%, lost on a fake last play {:.2}%, lost otherwise {:.2}%); logs: won+lost-on-fake {:.2}%",
            (d[0] + d[1] + d[2]) as f64 / r * 100.0, d[0] as f64 / r * 100.0, d[1] as f64 / r * 100.0, d[2] as f64 / r * 100.0, (d[0] + d[1]) as f64 / r * 100.0);
    }
    println!("claims {}  (unforced {})", rows.len(), unforced.len());
    let sz = |s: u8| rows.iter().filter(|r| r.0 == s).count() as f64 / rows.len() as f64;
    println!("claim size  1: {:.2}  2: {:.2}  3: {:.2}", sz(1), sz(2), sz(3));
    let groups: [(&str, fn(&Row) -> bool); 13] = [
        ("all", |_| true), ("left 0", |r| r.1 == 0), ("left 1-2", |r| (1..=2).contains(&r.1)), ("left 3-4", |r| r.1 >= 3),
        ("opening", |r| r.3 == 1), ("size 1", |r| r.0 == 1), ("size 2", |r| r.0 == 2), ("size 3", |r| r.0 == 3),
        ("4 alive", |r| r.2 == 4), ("3 alive", |r| r.2 == 3), ("2 alive", |r| r.2 == 2),
        ("total 7+", |r| r.7 >= 7), ("t7+ left0", |r| r.7 >= 7 && r.1 == 0)];
    for (name, f) in groups {
        let u: Vec<&&Row> = unforced.iter().filter(|r| f(r)).collect();
        let c: Vec<&Row> = rows.iter().filter(|r| f(r) && r.5).collect();
        let cr = u.iter().filter(|r| r.5).count() as f64 / u.len().max(1) as f64;
        let lr = c.iter().filter(|r| r.6).count() as f64 / c.len().max(1) as f64;
        println!("  {name:<9} call rate {cr:.2} (n={:5})   lie when called {lr:.2} (n={:5})", u.len(), c.len());
    }
    // Where late lies come from: plays made with no real card in hand (forced faking) vs by choice.
    for (name, lo, hi) in [("left 1-2", 1u8, 2u8), ("left 0", 0, 0), ("left 3-4", 3, 4)] {
        let all: Vec<&Row> = rows.iter().filter(|r| r.1 >= lo && r.1 <= hi).collect();
        let none = all.iter().filter(|r| !r.8).count();
        let lies_none = all.iter().filter(|r| !r.8 && r.6).count();
        let lies_had = all.iter().filter(|r| r.8 && r.6).count();
        let had = all.len() - none;
        println!("  {name:<9} plays {:6}: no real card {:.0}% (lies {:.0}% of those)   had a real card: lies {:.0}%",
            all.len(), 100.0 * none as f64 / all.len().max(1) as f64, 100.0 * lies_none as f64 / none.max(1) as f64, 100.0 * lies_had as f64 / had.max(1) as f64);
    }
}


/// Research harness (2026-11-06): a saved network (plain or transformer) in seat 0 against a
/// simulated population, optionally wrapped exactly as the live advisor wraps it.
/// liars fiteval --net <base> [--opp rankedfit|ranked|human] [--players 4|3|2] [--games N] [--threads T]
///               [--rules] [--nocall] [--callthresh X] [--greedy] [--seed S]
fn cmd_fiteval(a: &Args) {
    use std::sync::atomic::AtomicU64;
    let net_path = a.str("net", "C:/liars/nn/devil_tnet_r1v1");
    let net = Arc::new(nn::Model::load(&net_path).expect("loading --net"));
    let games = a.get("games", 10_000usize);
    let threads = a.get("threads", 4usize);
    let players = a.get("players", 4usize);
    let opp = a.str("opp", "rankedfit");
    let rules = a.flag("rules");
    let nocall = a.flag("nocall");
    let greedy = a.flag("greedy");
    let callthresh: Option<f64> = a.map.get("callthresh").and_then(|v| v.parse().ok());
    let datacall: Option<Arc<datacall::DataCall>> = a.map.get("datacall").map(|p| {
        let p = if p.is_empty() || p == "1" { "C:/liars/research/call_model_v61.json" } else { p.as_str() };
        Arc::new(datacall::DataCall::load(p).unwrap_or_else(|e| panic!("{e}")))
    });
    let seed = a.get("seed", 777u64);
    let stats: Arc<[AtomicU64; 3]> = Arc::new([AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)]);
    // (won, died first, my claims called, of which lies, rounds, my pulls by cause, others' pulls, number of others)
    let res: Vec<(bool, bool, u64, u64, u64, [u64; 4], u64, u64, [[u64; 2]; 12])> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads).map(|t| {
            let net = net.clone();
            let opp = opp.clone();
            let stats = stats.clone();
            let datacall = datacall.clone();
            sc.spawn(move || {
                let mut rng = Rng::new(seed + t as u64);
                let mut out = Vec::new();
                for _ in (t..games).step_by(threads) {
                    let mut ag: Vec<Box<dyn agents::Agent>> = (0..4).map(|q| if q == 0 {
                        {
                            let mut rw = agents::RuleWrap::new(net.clone(), rules, nocall, callthresh, greedy, stats.clone());
                            rw.datacall = datacall.clone();
                            Box::new(rw) as Box<dyn agents::Agent>
                        }
                    } else {
                        match opp.as_str() {
                            "rankedfit" => Box::new(agents::FitHuman::random(&mut rng)) as Box<dyn agents::Agent>,
                            "ranked" => Box::new(agents::HumanAgent { ranked: true, ..agents::HumanAgent::random_traits(0.4 + 2.1 * rng.f64() * rng.f64(), &mut rng) }),
                            k => agents::make_agent(k, None),
                        }
                    }).collect();
                    let mut g = if players < 4 {
                        let mut alive = [false; engine::MAX_PLAYERS];
                        alive[0] = true;
                        let mut others = vec![1usize, 2, 3];
                        for _ in 1..players {
                            let i = rng.below(others.len() as u32) as usize;
                            alive[others.swap_remove(i)] = true;
                        }
                        let mut shots = [0u8; engine::MAX_PLAYERS];
                        for q in 0..4 { if alive[q] { shots[q] = rng.below(4) as u8; } }
                        let live: Vec<usize> = (0..4).filter(|&q| alive[q]).collect();
                        let starter = live[rng.below(live.len() as u32) as usize];
                        engine::Game::new_mode(4, alive, shots, starter, true, &mut rng)
                    } else {
                        engine::Game::new_game(4, true, &mut rng)
                    };
                    g.ranked = true;
                    g.record = true;
                    g.net_seat[0] = true;
                    let na = g.n_alive() as u8;
                    g.log_ev(engine::Ev { kind: 5, actor: g.starter as u8, n: na, ..Default::default() });
                    let alive_start = g.n_alive();
                    let mut died_first: Option<bool> = None;
                    let (mut called, mut called_lie, mut rounds) = (0u64, 0u64, 0u64);
                    // Hero pulls by cause: caught lying, own call wrong (voluntary), forced call on a true claim, other (Devil / deal).
                    let mut cause = [0u64; 4];
                    let mut others_pulls = 0u64;
                    let mut mix = [[0u64; 2]; 12];
                    let mut pending_mix: Option<usize> = None;
                    let shots0: Vec<u8> = (0..4).map(|q| g.shots[q]).collect();
                    while !g.game_over {
                        let p = g.current;
                        let before: Vec<u8> = (0..4).map(|q| g.shots[q]).collect();
                        let mut ctx: Option<usize> = None;
                        if g.deal_legal() && ag[p].decide_deal(&g, p, &mut rng) {
                            g.apply_deal(&mut rng);
                            ctx = Some(3);
                        } else if g.is_forced(p) || (g.last.is_some() && ag[p].decide_call(&g, p, &mut rng)) {
                            let forced = g.is_forced(p);
                            if let Some(k) = pending_mix.take() { mix[k][1] += 1; }
                            if let Some(l) = g.last { if l.player == 0 { called += 1; if !l.devil && l.n_true < l.n { called_lie += 1; } } }
                            let claimer = g.last.map(|l| l.player);
                            g.apply_call();
                            ctx = Some(if claimer == Some(0) { 0 } else if p == 0 && !forced { 1 } else if p == 0 { 2 } else { 3 });
                        } else {
                            pending_mix = None;
                            let act = ag[p].decide_play(&g, p, &mut rng);
                            g.apply_action(act);
                            if p == 0 && act != engine::DEVIL_PLAY {
                                let n = (act.0 + act.1) as usize;
                                let k = (n - 1) * 4 + (g.hand(0) == 0) as usize * 2 + (act.1 > 0) as usize;
                                mix[k][0] += 1;
                                pending_mix = Some(k);
                            }
                        }
                        if g.round_over {
                            pending_mix = None;
                            rounds += 1;
                            g.finish_round(StarterRule::Loser, &mut rng);
                            if died_first.is_none() && g.n_alive() < alive_start {
                                died_first = Some(!g.alive[0]);
                            }
                        }
                        if g.shots[0] > before[0] {
                            let c = ctx.unwrap_or(3);
                            // A pull caused by a Devil reveal or a deal is "other" even when the hero was the claimer.
                            cause[if g.devil_reveal || g.deal_called { 3 } else { c }] += (g.shots[0] - before[0]) as u64;
                        }
                        for q in 1..4 { if g.shots[q] > before[q] { others_pulls += (g.shots[q] - before[q]) as u64; } }
                    }
                    let _ = shots0;
                    out.push((g.winner == 0, died_first == Some(true), called, called_lie, rounds, cause, others_pulls, alive_start as u64 - 1, mix));
                }
                out
            })
        }).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let n = res.len() as f64;
    let n_games = n;
    let wr = res.iter().filter(|r| r.0).count() as f64 / n;
    let df = res.iter().filter(|r| r.1).count() as f64 / n;
    let called: u64 = res.iter().map(|r| r.2).sum();
    let called_lie: u64 = res.iter().map(|r| r.3).sum();
    let rounds: u64 = res.iter().map(|r| r.4).sum();
    let ci = 1.96 * (wr * (1.0 - wr) / n).sqrt();
    let (plays, bluffs, forced) = (stats[0].load(Relaxed), stats[1].load(Relaxed), stats[2].load(Relaxed));
    println!("fiteval net={net_path} opp={opp} players={players} rules={rules} nocall={nocall} callthresh={callthresh:?} datacall={} greedy={greedy} games={}", datacall.is_some(), res.len());
    println!("  win rate {wr:.3} +/- {ci:.3}   died first {df:.3}   rounds/game {:.1}", rounds as f64 / n);
    println!("  my plays {plays}: bluff rate {:.2}, rule-forced {:.2}; my claims called {:.2}/game, of which lies {:.0}%",
        bluffs as f64 / plays.max(1) as f64, forced as f64 / plays.max(1) as f64, called as f64 / n, 100.0 * called_lie as f64 / called.max(1) as f64);
    let cause: Vec<f64> = (0..4).map(|i| res.iter().map(|r| r.5[i]).sum::<u64>() as f64 / n).collect();
    let others: f64 = res.iter().map(|r| r.6 as f64 / r.7.max(1) as f64).sum::<f64>() / n;
    println!("  my pulls/game {:.2} = caught lying {:.2} + my call wrong {:.2} + forced call true {:.2} + devil/deal {:.2};  other players {:.2} pulls/game each",
        cause.iter().sum::<f64>(), cause[0], cause[1], cause[2], cause[3], others);
    println!("  claim mix (per game: claims, called share):");
    for n in 1..=3 {
        let mut parts = Vec::new();
        for (left0, lie) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            let k = (n - 1) * 4 + left0 * 2 + lie;
            let c: u64 = res.iter().map(|r| r.8[k][0]).sum();
            let cc: u64 = res.iter().map(|r| r.8[k][1]).sum();
            parts.push(format!("{}{}: {:.2} ({:.0}%)", if left0 == 1 { "empty " } else { "" }, if lie == 1 { "lie" } else { "true" }, c as f64 / n_games, 100.0 * cc as f64 / c.max(1) as f64));
        }
        println!("    size {n}: {}", parts.join("  |  "));
    }
}

fn cmd_inspect(a: &Args) {
    let policy = load_policy(a);
    let n = a.get("players", 4usize);
    let games = a.get("games", 50_000u64);
    let rule = rule_of(a);
    let (c, p) = policy.trained_counts();
    println!("policy: {} rounds | info sets {} call, {} play", fmt_n(policy.iterations), fmt_n(c as u64), fmt_n(p as u64));
    println!("measured over {} self-play games, using the policy's call probability at every reached decision\n", fmt_n(games));

    // Call rate by how plausible the claimer's story is and by revolver matchup.
    let rows = ["claimer story impossible", "P(true) < 0.2", "0.2 - 0.4", "0.4 - 0.6", "0.6 - 0.8", "P(true) >= 0.8"];
    let cols = ["both 0-1 pulls", "me 0-1, them 3+", "me 3+, them 0-1", "both 3+"];
    let mut sum = [[0.0f64; 4]; 6];
    let mut cnt = [[0u64; 4]; 6];
    // Bluff rates by matching cards held.
    let mut bluff = [(0.0f64, 0u64); 6];
    let mut rng = Rng::new(a.get("seed", 99u64));
    let mut agents: Vec<Box<dyn agents::Agent>> = (0..n).map(|_| Box::new(agents::CfrAgent::new(policy.clone())) as Box<dyn agents::Agent>).collect();
    for _ in 0..games {
        let mut g = engine::Game::new_game(n, policy.devil, &mut rng);
        while !g.game_over {
            let p = g.current;
            if g.is_forced(p) {
                g.apply_call();
            } else {
                if let Some(lp) = g.last {
                    if let Some(pr) = policy.call_probs(policy::call_key(&g, p, &policy.feat())) {
                        let unseen = engine::MATCHING_IN_DECK - g.hm[p] - g.played_match[p];
                        let r = if g.claims[lp.player] > unseen {
                            0
                        } else {
                            1 + ((agents::p_claim_true(&g, p) * 5.0) as usize).min(4)
                        };
                        let (me, them) = (g.shots[p], g.shots[lp.player]);
                        let col = match (me >= 3, them >= 3) {
                            (false, false) if me <= 1 && them <= 1 => Some(0),
                            (false, true) if me <= 1 => Some(1),
                            (true, false) if them <= 1 => Some(2),
                            (true, true) => Some(3),
                            _ => None,
                        };
                        if let Some(col) = col {
                            sum[r][col] += pr[1] as f64;
                            cnt[r][col] += 1;
                        }
                    }
                }
                if g.last.is_some() && agents[p].decide_call(&g, p, &mut rng) {
                    g.apply_call();
                } else {
                    let m = g.hm[p] as usize;
                    let lg = &LEGAL[m][g.hn[p] as usize];
                    if let Some(pr) = policy.play_probs(policy::play_key(&g, p, &policy.feat())) {
                        let pb: f64 = (0..lg.n).filter(|&i| lg.acts[i].1 > 0).map(|i| pr[i] as f64).sum();
                        bluff[m].0 += pb;
                        bluff[m].1 += 1;
                    }
                    let act = agents[p].decide_play(&g, p, &mut rng);
                    g.apply_action(act);
                }
            }
            if g.round_over {
                g.finish_round(rule, &mut rng);
            }
        }
    }
    println!("P(call), 'P(true)' = hypergeometric chance the claimer's claims this round are all true");
    println!("{:<26}{}", "", cols.iter().map(|c| format!("{c:>17}")).collect::<String>());
    for (r, name) in rows.iter().enumerate() {
        let cells: String = (0..4)
            .map(|col| if cnt[r][col] >= 50 { format!("{:>9.2} ({:>5})", sum[r][col] / cnt[r][col] as f64, short(cnt[r][col])) } else { format!("{:>17}", "-") })
            .collect();
        println!("{name:<26}{cells}");
    }
    println!("\nP(play includes a bluff card) by matching cards in hand");
    for (m, (s, k)) in bluff.iter().enumerate() {
        if *k > 0 {
            println!("  {m} matching: {:.2}  ({} decisions)", s / *k as f64, short(*k));
        }
    }
    for (me, nxt) in [(0u64, 0u64), (0, 4), (4, 0)] {
        println!("\nOpening play | {n} alive, next player has 5 cards | me {me} pulls, next {nxt} pulls");
        for m in 0..=5u64 {
            let nm = 5 - m;
            let slack = (8 - m as i32).clamp(-1, 6) + 1;
            let k = play_index([n as u64 - 2, m, nm, 5, me, nxt, slack as u64, 0, (n as u64 - 2).min(2), 0, 0, 0, 0]);
            let lg = &LEGAL[m as usize][nm as usize];
            match policy.play_probs(k) {
                Some(pr) => {
                    let mut v: Vec<((u8, u8), f32)> = (0..lg.n).map(|i| (lg.acts[i], pr[i])).collect();
                    v.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
                    let desc: Vec<String> = v.iter().take(4).map(|((t, b), p)| format!("{t} true + {b} bluff: {p:.2}")).collect();
                    println!("{m} matching: {}", desc.join(", "));
                }
                None => println!("{m} matching: -"),
            }
        }
    }
}

fn short(k: u64) -> String {
    if k >= 1_000_000 { format!("{}M", k / 1_000_000) } else if k >= 1000 { format!("{}k", k / 1000) } else { k.to_string() }
}

/// Hero wrapper for `spot`: at the first "no true cards, believable claim" call decision it
/// takes the forced arm (call or not), then plays the policy as usual.
struct SpotHero {
    inner: agents::CfrAgent,
    call: bool,
    min_true: f64,
    hit: bool,
}

impl agents::Agent for SpotHero {
    fn decide_call(&mut self, g: &engine::Game, p: usize, rng: &mut Rng) -> bool {
        if !self.hit && g.hm[p] == 0 && g.hd[p] == 0 && agents::p_claim_true(g, p) >= self.min_true {
            self.hit = true;
            return self.call;
        }
        self.inner.decide_call(g, p, rng)
    }
    fn decide_play(&mut self, g: &engine::Game, p: usize, rng: &mut Rng) -> (u8, u8) {
        self.inner.decide_play(g, p, rng)
    }
    fn as_any_hit(&self) -> Option<bool> {
        Some(self.hit)
    }
}

/// Call vs bluff in the "no true cards, claim looks true" spot: each game is played twice from the
/// same seed, once calling and once not, so the only difference is that one decision.
fn cmd_spot(a: &Args) {
    let policy = load_policy(a);
    let n = a.get("players", 4usize);
    let games = a.get("games", 200_000u64);
    let threads = a.get("threads", 8usize);
    let min_true = a.get("min-true", 0.7f64);
    let rule = rule_of(a);
    println!("spot: no true cards in hand, claim at least {:.0}% likely true (card count) | {} games per opponent, {threads} threads",
        min_true * 100.0, fmt_n(games));
    for opp in ["human", "prob", "self"] {
        let per = games / threads as u64;
        let results: Vec<(u64, u64, u64)> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|th| {
                    let policy = policy.clone();
                    s.spawn(move || {
                        let (mut hits, mut win_call, mut win_bluff) = (0u64, 0u64, 0u64);
                        for gi in 0..per {
                            let seed = (th as u64) << 40 | gi;
                            let seat = (gi % n as u64) as usize;
                            let mut won = [false; 2];
                            let mut hit = false;
                            for (arm, call) in [true, false].into_iter().enumerate() {
                                let mut rng = Rng::new(seed);
                                let mut ag: Vec<Box<dyn agents::Agent>> = (0..n)
                                    .map(|s| -> Box<dyn agents::Agent> {
                                        if s == seat {
                                            Box::new(SpotHero { inner: agents::CfrAgent::new(policy.clone()), call, min_true, hit: false })
                                        } else {
                                            agents::make_agent(if opp == "self" { "policy" } else { opp }, Some(&policy))
                                        }
                                    })
                                    .collect();
                                let w = agents::play_game(&mut ag, &mut rng, rule, policy.devil, |_| {});
                                hit |= ag[seat].as_any_hit().unwrap_or(false);
                                won[arm] = w == seat;
                            }
                            if hit {
                                hits += 1;
                                win_call += won[0] as u64;
                                win_bluff += won[1] as u64;
                            }
                        }
                        (hits, win_call, win_bluff)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let (hits, wc, wb) = results.iter().fold((0, 0, 0), |a, r| (a.0 + r.0, a.1 + r.1, a.2 + r.2));
        let h = hits.max(1) as f64;
        println!("vs {opp:<6} spot reached in {} games | win rate if you CALL {:.3} | if you BLUFF/PLAY {:.3}",
            fmt_n(hits), wc as f64 / h, wb as f64 / h);
    }
}

/// liars merge [--chaos] [--table-log N] BASE OUT WORKER...
/// Combines checkpoints of independent runs that all resumed from BASE: OUT = BASE + sum(WORKER - BASE).
fn cmd_merge(raw: &[String]) {
    let chaos = raw.iter().any(|a| a == "--chaos");
    let mut log2 = 23u32;
    let mut files: Vec<&String> = Vec::new();
    let mut i = 0;
    while i < raw.len() {
        match raw[i].as_str() {
            "--chaos" => {}
            "--table-log" => {
                i += 1;
                log2 = raw[i].parse().expect("--table-log N");
            }
            _ => files.push(&raw[i]),
        }
        i += 1;
    }
    assert!(files.len() >= 3, "usage: liars merge [--chaos] [--table-log N] BASE OUT WORKER...");
    let (base_path, out_path, workers) = (files[0], files[1], &files[2..]);
    if chaos {
        let base = chaos::ChaosTables::new(log2);
        let merged = chaos::ChaosTables::new(log2);
        let (t0, n) = base.load(base_path).expect("base");
        merged.load(base_path).expect("base");
        let mut t = t0;
        for w in workers {
            let wt = chaos::ChaosTables::new(log2);
            let (ti, _) = wt.load(w).unwrap_or_else(|e| panic!("{w}: {e}"));
            for k in 0..3 {
                merged.t[k].add_delta(&wt.t[k], &base.t[k]);
            }
            t += ti - t0;
        }
        merged.t.iter().for_each(|x| x.clamp_regrets());
        merged.save(out_path, t, n).expect("save");
        println!("merged {} Chaos workers: {} -> {} rounds", workers.len(), fmt_n(t0), fmt_n(t));
    } else {
        let base = Tables::new(log2);
        let merged = Tables::new(log2);
        let (t0, n) = base.load_checkpoint(base_path).expect("base");
        merged.load_checkpoint(base_path).expect("base");
        let mut t = t0;
        for w in workers {
            let wt = Tables::new(log2);
            let (ti, _) = wt.load_checkpoint(w).unwrap_or_else(|e| panic!("{w}: {e}"));
            merged.call.add_delta(&wt.call, &base.call);
            merged.play.add_delta(&wt.play, &base.play);
            t += ti - t0;
        }
        merged.call.clamp_regrets();
        merged.play.clamp_regrets();
        merged.save_checkpoint(out_path, t, n).expect("save");
        println!("merged {} workers: {} -> {} rounds", workers.len(), fmt_n(t0), fmt_n(t));
    }
}

/// liars export --policy-from CHECKPOINT --out POLICY [--devil] [--chaos] [--table-log N]
/// Writes the average-strategy policy file from a checkpoint (e.g. after a merge).
fn cmd_export(a: &Args) {
    let ck = a.str("from", "checkpoint.bin");
    let out = a.str("out", "policy.bin");
    let log2 = a.get("table-log", 23u32);
    if a.flag("chaos") {
        let t = chaos::ChaosTables::new(log2);
        let (it, _) = t.load(&ck).expect("checkpoint");
        chaos::ChaosPolicy::from_tables(&t, it).save(&out).expect("save");
    } else {
        let t = Tables::new(log2);
        let (it, _) = t.load_checkpoint(&ck).expect("checkpoint");
        let f = policy::Feat {
            history: !a.flag("no-history"),
            self_image: a.flag("self-image"),
            coarse_calls: a.flag("coarse-calls"),
            claimer_calls: !a.flag("no-claimer-calls"),
            call_accuracy: !a.flag("no-call-accuracy"),
            after_next: a.flag("after-next"),
            threat: a.flag("threat"),
            full_table: a.flag("full-table"),
            fast_reads: a.flag("fast-reads"),
            call_table: a.flag("call-table"),
        };
        Policy::from_tables(&t, it, &f, a.flag("devil")).save(&out).expect("save");
    }
    println!("exported {out} from {ck}");
}

/// liars gen-data --policy P --samples N --out FILE [--threads T] [--seed S]
/// Records the champion policy's decisions (encoded for the network) from simulated games.
fn cmd_gen_data(a: &Args) {
    let policy = load_policy(a);
    let devil = policy.devil;
    let samples = a.get("samples", 1_000_000usize);
    let threads = a.get("threads", 8usize);
    let out = a.str("out", "nn_data.bin");
    let seed = a.get("seed", 1u64);
    let start = Instant::now();
    let per = samples / threads;
    let written: usize = std::thread::scope(|s| {
        let hs: Vec<_> = (0..threads).map(|t| {
            let pol = policy.clone();
            let path = format!("{out}.{t}");
            s.spawn(move || nn::generate(pol, devil, per, seed * 1000 + t as u64, &path).expect("write"))
        }).collect();
        hs.into_iter().map(|h| h.join().unwrap()).sum()
    });
    println!("wrote {} samples ({} floats each) to {out}.0..{} in {:.0}s", fmt_n(written as u64), nn::ROW, threads - 1, start.elapsed().as_secs_f64());
}

fn cmd_bench(a: &Args) {
    let n = a.get("players", 4usize);
    let rounds = a.get("rounds", 2_000_000u64);
    let trainer = Trainer {
        tables: Tables::new(20),
        t: AtomicU64::new(0),
        n_players: n,
        epsilon: 0.6,
        rule: rule_of(a),
        value: ValueModel::new(n),
        starts: StartSampler::prior(n),
        bot_share: 0.5,
        full_game_share: 0.5,
        history: true,
        vary_bots: true,
        devil: a.flag("devil"),
        feat: policy::Feat::full(true, false),
    };
    let mut rng = Rng::new(1);
    let mut recs = Vec::with_capacity(64);
    let mut bots = Bots::default();
    let start = Instant::now();
    while trainer.t.load(Relaxed) < rounds {
        trainer.iterate(&mut rng, &mut recs, &mut bots);
    }
    let dt = start.elapsed().as_secs_f64();
    println!("{} rounds in {dt:.2}s = {} rounds/s on one thread", fmt_n(trainer.t.load(Relaxed)), fmt_n((trainer.t.load(Relaxed) as f64 / dt) as u64));
}

fn main() {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.is_empty() {
        print!("{USAGE}");
        return;
    }
    let args = Args::parse(&raw[1..]);
    match raw[0].as_str() {
        "train" => cmd_train(&args),
        "value" => cmd_value(&args),
        "eval" => cmd_eval(&args),
        "fiteval" => cmd_fiteval(&args),
        "h2h" => cmd_h2h(&args),
        "simstats" => cmd_simstats(&args),
        "inspect" => cmd_inspect(&args),
        "bench" => cmd_bench(&args),
        "spot" => cmd_spot(&args),
        "merge" => cmd_merge(&raw[1..]),
        "export" => cmd_export(&args),
        "gen-data" => cmd_gen_data(&args),
        "watch" => {
            ui::enable_colors();
            let devil_path = args.str("devil-policy", "devil.bin");
            let devil = Policy::load(&devil_path).ok().map(Arc::new);
            println!("Devil policy: {}", if devil.is_some() { devil_path.as_str() } else { "not found (Devil advice disabled)" });
            let chaos_path = args.str("chaos-policy", "chaos.bin");
            let chaos = chaos::ChaosPolicy::load(&chaos_path).ok().map(Arc::new);
            println!("Chaos policy: {}", if chaos.is_some() { chaos_path.as_str() } else { "not found (Chaos advice uses a fallback)" });
            watch::run(load_policy(&args), devil, chaos, &args.str("logs", r"C:\liars\logs"), args.flag("once"), args.flag("auto"))
        }
        _ => print!("{USAGE}"),
    }
}
