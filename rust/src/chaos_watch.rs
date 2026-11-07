//! Live advisor for Chaos games: rebuilds the public state from the LiarsTells log and shows the
//! Chaos policy's call / play odds on my turn and its target odds when I get to aim.

use std::sync::Arc;

use serde_json::Value;

use crate::chaos::{call_key, play_key, target_choices, target_key, CAgent, CGame, CProb, ChaosPolicy, Shot, CHAOS, KING, MASTER, QUEEN};
use crate::engine::MAX_PLAYERS;
use crate::rng::Rng;
use crate::ui::{bar_line, rule, BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RESET, YELLOW};

/// Game card codes: 1 King, 2 Queen, 3 Chaos, 4 Master.
fn kind_of(code: i64) -> Option<usize> {
    match code {
        1 => Some(KING),
        2 => Some(QUEEN),
        3 => Some(CHAOS),
        4 => Some(MASTER),
        _ => None,
    }
}

fn code_name(code: i64) -> &'static str {
    match code {
        1 => "K",
        2 => "Q",
        3 => "Chaos",
        4 => "Master",
        _ => "?",
    }
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default()
}

#[derive(Default)]
pub struct ChaosWatch {
    policy: Option<Arc<ChaosPolicy>>,
    rng: Option<Rng>,
    names: Vec<String>,
    slots: Vec<i64>,
    /// Log seat index -> player (players are identified by name and seat slot).
    raw: Vec<Option<usize>>,
    me: usize,
    dir_votes: i32,
    prev_turn: Option<usize>,
    alive: Vec<bool>,
    shots: Vec<u8>,
    hands: Vec<u8>,
    claims: Vec<u8>,
    claims_total: u8,
    last: Option<usize>,
    hand: Vec<i64>,
    table: i64,
    /// Cards I have played since the last full deal, by kind.
    played: [u8; 4],
    chaos_out: bool,
    called: Vec<u8>,
    lies: Vec<u8>,
    opps: Vec<u8>,
    calls: Vec<u8>,
    in_round: bool,
    f_liar: bool,
    f_chaos: bool,
    f_master: bool,
    f_caught: bool,
    advised_aim: bool,
}

impl ChaosWatch {
    pub fn new(policy: Option<Arc<ChaosPolicy>>) -> Self {
        ChaosWatch { policy, rng: Some(Rng::new(0xC4A05)), ..Default::default() }
    }

    fn n(&self) -> usize {
        self.names.len()
    }

    fn name(&self, p: usize) -> String {
        let n = self.names.get(p).cloned().unwrap_or_else(|| format!("seat {p}"));
        if p == self.me { format!("{n} (you)") } else { n }
    }

    fn player(&self, raw: i64) -> Option<usize> {
        if raw < 0 { None } else { self.raw.get(raw as usize).copied().flatten() }
    }

    fn grow(&mut self, n: usize) {
        for v in [&mut self.shots, &mut self.hands, &mut self.claims, &mut self.called, &mut self.lies, &mut self.opps, &mut self.calls] {
            if v.len() < n {
                v.resize(n, 0);
            }
        }
        if self.alive.len() < n {
            self.alive.resize(n, true);
        }
        if self.slots.len() < n {
            self.slots.resize(n, -1);
        }
        while self.names.len() < n {
            self.names.push(format!("seat {}", self.names.len()));
        }
    }

    fn turn_order(&self) -> Vec<usize> {
        let mut o: Vec<usize> = (0..self.n()).collect();
        o.sort_by_key(|&q| self.slots[q]);
        if self.dir_votes < 0 {
            o.reverse();
        }
        o
    }

    pub fn handle(&mut self, e: &Value) {
        let kind = e["e"].as_str().unwrap_or("");
        let raw = e["p"].as_i64().unwrap_or(-1);
        if kind == "seat" {
            self.on_seat(raw, e);
            return;
        }
        if kind == "round" {
            self.on_round(e);
            return;
        }
        if kind == "cflags" {
            self.on_flags(e);
            return;
        }
        let Some(p) = self.player(raw) else { return };
        match kind {
            "turn" => self.on_turn(p, e),
            "play" => self.on_play(p, e),
            "pull" => self.on_pull(e),
            "aim" => self.on_aim(p, e),
            "chamber" => self.shots[p] = e["to"].as_u64().unwrap_or(0).min(5) as u8,
            "tamper" => println!("  {RED}{BOLD}⚠ CHEATING? {}{RESET}: {}", self.name(p), e["what"].as_str().unwrap_or("?")),
            "dead" => {
                self.alive[p] = false;
                println!("  {RED}{BOLD}{} is dead{RESET}", self.name(p));
            }
            _ => {}
        }
    }

    fn on_seat(&mut self, raw: i64, e: &Value) {
        if raw < 0 {
            return;
        }
        let name = e["name"].as_str().unwrap_or("?").to_string();
        let slot = e["slot"].as_i64().unwrap_or(-1);
        let known = (0..self.n()).find(|&q| self.names[q] == name && self.slots[q] == slot);
        let q = known.unwrap_or_else(|| {
            let q = self.n();
            self.grow(q + 1);
            self.names[q] = name.clone();
            self.slots[q] = slot;
            q
        });
        let r = raw as usize;
        if self.raw.len() <= r {
            self.raw.resize(r + 1, None);
        }
        self.raw[r] = Some(q);
        if e["me"].as_bool() == Some(true) {
            self.me = q;
        }
        if known.is_none() {
            println!("seat {q}: {name}{}", if q == self.me && e["me"].as_bool() == Some(true) { " (you)" } else { "" });
        }
    }

    fn on_round(&mut self, e: &Value) {
        let new_hand = ints(&e["hand"]);
        if new_hand.len() > self.hand.len() || !self.in_round && self.claims_total == 0 && self.played == [0; 4] {
            // A fresh deal: everything played before is back in the deck.
            self.played = [0; 4];
            self.chaos_out = false;
        }
        self.hand = new_hand;
        self.table = e["table"].as_i64().unwrap_or(1);
        for (i, &h) in ints(&e["hands"]).iter().enumerate() {
            if let Some(q) = self.player(i as i64) {
                self.hands[q] = h as u8;
            }
        }
        for (i, &v) in ints(&e["pulls"]).iter().enumerate() {
            if let Some(q) = self.player(i as i64) {
                self.shots[q] = (v as u8).min(5);
            }
        }
        // Death is final within a game: the game can reset a dead (often departed) player's seat
        // to "alive" with 0 pulls. Only a genuinely new game (everyone alive, all pulls 0) revives.
        let new_game = ints(&e["pulls"]).iter().all(|&v| v == 0) && ints(&e["alive"]).iter().all(|&v| v != 0);
        for (i, &v) in ints(&e["alive"]).iter().enumerate() {
            if let Some(q) = self.player(i as i64) {
                self.alive[q] = v != 0 && (new_game || self.alive[q]);
            }
        }
        self.claims.iter_mut().for_each(|c| *c = 0);
        self.claims_total = 0;
        self.last = None;
        self.prev_turn = None;
        self.in_round = true;
        let hand: Vec<&str> = self.hand.iter().map(|&c| code_name(c)).collect();
        println!();
        println!("{CYAN}--- Chaos round {} | table card {BOLD}{}{RESET}{CYAN} | your hand: {} ---{RESET}", e["n"], code_name(self.table), hand.join(" "));
        for q in 0..self.n() {
            if self.alive[q] {
                let rec = if self.called[q] > 0 { format!(", caught lying {}/{}", self.lies[q], self.called[q]) } else { String::new() };
                println!("    {:<20} {} cards, revolver {}/6{rec}", self.name(q), self.hands[q], self.shots[q]);
            }
        }
    }

    fn on_flags(&mut self, e: &Value) {
        let liar = e["liar"].as_bool().unwrap_or(false);
        let chaos = e["chaos"].as_bool().unwrap_or(false);
        let master = e["master"].as_bool().unwrap_or(false);
        let caught = e["caught"].as_bool().unwrap_or(false);
        if chaos && !self.f_chaos {
            // The last "play" was the Chaos card: it can't be called, and everyone aims now.
            if let Some(p) = self.last.take() {
                println!("  {MAGENTA}{BOLD}CHAOS!{RESET} {} threw the Chaos card: everyone shoots someone", self.name(p));
            } else {
                println!("  {MAGENTA}{BOLD}CHAOS!{RESET} everyone shoots someone");
            }
            self.chaos_out = true;
            self.advised_aim = false;
        }
        if master && !self.f_master {
            println!("  {YELLOW}{BOLD}MASTER revealed{RESET}: its thrower picks someone to shoot");
            self.advised_aim = false;
        }
        if caught && !self.f_caught {
            println!("  {YELLOW}{BOLD}Lie caught{RESET}: the caller has to shoot the liar");
        }
        self.f_liar = liar;
        self.f_chaos = chaos;
        self.f_master = master;
        self.f_caught = caught;
        if !chaos && !master {
            self.advised_aim = false;
        }
    }

    fn on_turn(&mut self, p: usize, e: &Value) {
        if let Some(prev) = self.prev_turn {
            if prev != p {
                let order = {
                    let mut o: Vec<usize> = (0..self.n()).collect();
                    o.sort_by_key(|&q| self.slots[q]);
                    o
                };
                let n = order.len();
                if let Some(i) = order.iter().position(|&s| s == prev) {
                    if order[(i + 1) % n] == p {
                        self.dir_votes += 1;
                    } else if order[(i + n - 1) % n] == p {
                        self.dir_votes -= 1;
                    }
                }
            }
        }
        self.prev_turn = Some(p);
        if p != self.me || !self.in_round {
            return;
        }
        if e.get("hand").is_some() {
            self.hand = ints(&e["hand"]);
        }
        self.advise_turn();
    }

    fn on_play(&mut self, p: usize, e: &Value) {
        if !self.in_round {
            return;
        }
        if self.last.is_some() {
            self.opps[p] = self.opps[p].saturating_add(1);
        }
        if p == self.me && e.get("hand").is_some() {
            let after = ints(&e["hand"]);
            for code in 1..=4 {
                let before = self.hand.iter().filter(|&&c| c == code).count();
                let now = after.iter().filter(|&&c| c == code).count();
                if before > now {
                    if let Some(k) = kind_of(code) {
                        self.played[k] += (before - now) as u8;
                    }
                }
            }
            self.hand = after;
        }
        self.claims[p] += 1;
        self.claims_total += 1;
        self.last = Some(p);
        self.hands[p] = e["left"].as_u64().unwrap_or(0) as u8;
        println!("  {BOLD}{}{RESET} claims {}  {DIM}({} left, {:.1}s){RESET}",
            self.name(p), code_name(self.table), self.hands[p], e["think"].as_f64().unwrap_or(-1.0));
    }

    fn on_pull(&mut self, e: &Value) {
        if !self.in_round {
            return;
        }
        self.in_round = false;
        let accused = self.player(e["accused"].as_i64().unwrap_or(-1));
        let caller = self.player(e["caller"].as_i64().unwrap_or(-1));
        let revealed = ints(&e["revealed"]);
        let master = revealed.contains(&4);
        let lie = e["lie"].as_bool().unwrap_or(false) && !master;
        if let Some(a) = accused {
            self.called[a] = self.called[a].saturating_add(1);
            if lie {
                self.lies[a] = self.lies[a].saturating_add(1);
            }
        }
        if let Some(c) = caller {
            self.opps[c] = self.opps[c].saturating_add(1);
            self.calls[c] = self.calls[c].saturating_add(1);
        }
        let rev: Vec<&str> = revealed.iter().map(|&c| code_name(c)).collect();
        let what = if master { "the MASTER" } else if lie { "a LIE" } else { "TRUE" };
        println!("  {YELLOW}{BOLD}LIAR!{RESET} {} called {}: it was {what} [{}]",
            caller.map_or("?".into(), |c| self.name(c)), accused.map_or("?".into(), |a| self.name(a)), rev.join(" "));
    }

    fn on_aim(&mut self, p: usize, e: &Value) {
        let on = e["on"].as_bool().unwrap_or(false);
        if on && p == self.me && !self.advised_aim && (self.f_chaos || self.f_master) {
            self.advised_aim = true;
            self.advise_target(if self.f_chaos { Shot::Chaos } else { Shot::Master });
        }
    }

    /// The public state as a Chaos game, seats renumbered into turn order. Other players' hands
    /// are card counts only.
    fn to_game(&mut self) -> (CGame, usize, Vec<usize>) {
        let order = self.turn_order();
        let n = order.len();
        let mut alive = [false; MAX_PLAYERS];
        let mut shots = [0u8; MAX_PLAYERS];
        for (k, &s) in order.iter().enumerate() {
            alive[k] = self.alive[s];
            shots[k] = self.shots[s].min(5);
        }
        let me = order.iter().position(|&s| s == self.me).unwrap_or(0);
        let rng = self.rng.as_mut().unwrap();
        let mut g = CGame::new(n, alive, shots, me, rng);
        g.table = if self.table == 2 { QUEEN } else { KING };
        for (k, &s) in order.iter().enumerate() {
            g.hand[k] = [0; 4];
            if k == me {
                for &c in &self.hand {
                    if let Some(kind) = kind_of(c) {
                        g.hand[k][kind] += 1;
                    }
                }
                g.played[k] = self.played;
            } else {
                g.hand[k][1 - g.table] = self.hands[s];
                g.played[k] = [0; 4];
            }
            g.claims[k] = self.claims[s];
            g.rec_called[k] = self.called[s];
            g.rec_lies[k] = self.lies[s];
            g.rec_opps[k] = self.opps[s];
            g.rec_calls[k] = self.calls[s];
        }
        g.claims_total = self.claims_total;
        g.chaos_out = self.chaos_out;
        g.last = self.last.and_then(|p| order.iter().position(|&s| s == p)).map(|k| (k, 1 - g.table));
        g.current = me;
        g.round_over = false;
        g.target = [None; MAX_PLAYERS];
        g.choosing = [false; MAX_PLAYERS];
        (g, me, order)
    }

    fn header(&self, title: &str) {
        let hand: Vec<String> = self.hand.iter().map(|&c| {
            let col = if c == 3 { MAGENTA } else if c == self.table || c == 4 { GREEN } else { RED };
            format!("{col}{}{RESET}", code_name(c))
        }).collect();
        println!();
        println!("{}", rule(title));
        println!("  table {BOLD}{}{RESET}   your hand  {}   {DIM}(Master always counts as true){RESET}", code_name(self.table), hand.join(" "));
    }

    fn advise_turn(&mut self) {
        if self.hand.is_empty() {
            return;
        }
        let (g, me, order) = self.to_game();
        self.header("YOUR TURN (Chaos)");
        if g.is_forced(me) {
            println!("  {YELLOW}{BOLD}You must call LIAR{RESET}  {DIM}(you are the only one with cards left){RESET}");
            println!("{}", rule(""));
            return;
        }
        let mut fallback = CProb::default();
        let mut rng = self.rng.take().unwrap();
        let rng = &mut rng;
        let mut call_p = 0.0;
        if let Some((lp, _)) = g.last {
            call_p = match self.policy.as_ref().and_then(|p| p.tables[0].get(&call_key(&g, me))) {
                Some(pr) => pr[1] as f64,
                None => if fallback.call(&g, me, rng) { 1.0 } else { 0.0 },
            };
            println!("  {BOLD}{}{RESET} just claimed {}  {DIM}(caught lie: you shoot them | true: you shoot yourself | Master: they shoot someone){RESET}",
                self.name(order[lp]), code_name(self.table));
            println!();
            println!("{}", bar_line(&format!("{YELLOW}{BOLD}{:<26}{RESET}", "CALL LIAR"), call_p, YELLOW));
        }
        let (ks, nk) = g.playable(me);
        let probs: Vec<f32> = match self.policy.as_ref().and_then(|p| p.tables[1].get(&play_key(&g, me))) {
            Some(pr) => pr[..nk].to_vec(),
            None => {
                let i = fallback.play(&g, me, rng);
                (0..nk).map(|j| if j == i { 1.0 } else { 0.0 }).collect()
            }
        };
        let scale = 1.0 - call_p;
        let mut idx: Vec<usize> = (0..nk).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        for i in idx {
            let p = probs[i] as f64 * scale;
            if p < 0.005 {
                continue;
            }
            let (label, col) = match ks[i] {
                k if k == g.table => (format!("PLAY [{}] truth", code_name(self.table)), GREEN),
                MASTER => ("PLAY [Master] truth".to_string(), GREEN),
                CHAOS => ("THROW [Chaos] all shoot".to_string(), MAGENTA),
                _ => (format!("PLAY [{}] bluff", code_name(3 - self.table)), RED),
            };
            println!("{}", bar_line(&format!("{col}{label:<26}{RESET}"), p, col));
        }
        println!("{}", rule(""));
        self.rng = Some(Rng::new(rng.next_u64()));
    }

    fn advise_target(&mut self, why: Shot) {
        let (g, me, order) = self.to_game();
        let choices = target_choices(&g, me);
        if choices.is_empty() {
            return;
        }
        self.header(if why == Shot::Chaos { "CHAOS: PICK A TARGET" } else { "MASTER: PICK A TARGET" });
        let mut fallback = CProb::default();
        let mut rng = self.rng.take().unwrap();
        let rng = &mut rng;
        let probs: Vec<f32> = match self.policy.as_ref().and_then(|p| p.tables[2].get(&target_key(&g, me, why))) {
            Some(pr) => pr[..choices.len()].to_vec(),
            None => {
                let i = fallback.target(&g, me, why, rng);
                (0..choices.len()).map(|j| if j == i { 1.0 } else { 0.0 }).collect()
            }
        };
        println!("  {DIM}your revolver: {}/6 (a hit kills them and reloads your gun){RESET}", g.shots[me]);
        let mut idx: Vec<usize> = (0..choices.len()).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        for i in idx {
            let q = choices[i];
            let label = format!("SHOOT {} ({} cards, {}/6)", self.names[order[q]], g.hand_size(q), g.shots[q]);
            println!("{}", bar_line(&format!("{RED}{label:<26}{RESET}"), probs[i] as f64, RED));
        }
        println!("{}", rule(""));
        self.rng = Some(Rng::new(rng.next_u64()));
    }
}
