//! Live advisor: follows the newest LiarsTells log, rebuilds the public game state and prints
//! what it thinks is happening, plus the policy's advice whenever it is my turn.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use crate::agents::{deal_p_all_lies, hypergeom_at_least, p_claim_true, Agent, ProbAgent};
use crate::engine::{Game, Play, DECK_SIZE, DEVIL_PLAY, HAND_SIZE, LEGAL, MATCHING_IN_DECK, MAX_PLAYERS};
use crate::policy::{call_key, deal_key, play_key, Policy};
use crate::chaos::ChaosPolicy;
use crate::chaos_watch::ChaosWatch;
use crate::rng::Rng;
use crate::ui::{bar_line, rule, BOLD, DIM, GREEN, MAGENTA, RED, RESET, YELLOW};

const JOKER: i64 = 4;
const DEVIL: i64 = -1;

fn card_name(c: i64) -> &'static str {
    match c {
        1 => "K",
        2 => "Q",
        3 => "A",
        4 => "Joker",
        -1 => "Devil",
        _ => "?",
    }
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default()
}

#[derive(Default, Clone)]
struct State {
    names: Vec<String>,
    me: usize,
    pos: Vec<(f64, f64)>,
    /// Seat numbers from the game (PlayerStats.Slot); -1 when unknown.
    slots: Vec<i64>,
    /// Times each player raised their cards since their turn started.
    looks: Vec<u8>,
    /// Card-check habit from past games, as the network reads it (rate x255, claims seen).
    habit: Vec<(u8, u8)>,
    /// Each player's usual think time: sum of ln(seconds) and count (seeded from past games).
    think_log: Vec<f64>,
    think_n: Vec<u32>,
    /// Public events this game for the transformer network (seats are State seats), and the
    /// index of this round's start event (its starter is filled in by the first play).
    evs: Vec<crate::engine::Ev>,
    round_ev: Option<usize>,
    /// Someone just called Devil's Deal (the plugin saw the game's banner): their seat. The next
    /// pull belongs to that deal, not to a Liar call.
    pending_deal: Option<usize>,
    /// Rounds logged in this game so far.
    round_no: u32,
    /// Times each player was passed over in turn order while holding cards (they left or are AFK).
    skipped: Vec<u8>,
    /// Pace label of the last claim ("fast" etc., relative to that player's usual).
    last_pace: Option<&'static str>,
    /// Cheat detection from public events: my hand at the deal, voluntary calls that were right,
    /// log-probability of having survived every pull so far, and tamper flags from the plugin.
    start_hand: Vec<i64>,
    /// This round's plays: (seat, cards, real cards, Devil); the last two only for mine.
    round_plays: Vec<(usize, u8, u8, bool)>,
    calls_right: Vec<u8>,
    calls_judged: Vec<u8>,
    /// Each player's habits so far (past games plus this one), from ProfTrack.
    hab: Vec<crate::engine::Prof>,
    luck_logp: Vec<f64>,
    flags: Vec<Vec<String>>,
    /// Log seat index -> player. Players are identified by name and seat slot, because log
    /// indexes can shift mid-game and the game can spawn a second object for one player.
    raw: Vec<Option<usize>>,
    dir_votes: i32,
    table: i64,
    hand: Vec<i64>,
    start_match: u8,
    hands: Vec<u8>,
    shots: Vec<u8>,
    alive: Vec<bool>,
    claims: Vec<u8>,
    claims_total: u8,
    last: Option<(usize, u8)>,
    /// The play before `last` (to recover the real accused when `last` was a forced caller's
    /// hand being cleared rather than a play).
    prev_last: Option<(usize, u8)>,
    called: Vec<u8>,
    lies: Vec<u8>,
    /// Times a call of mine on this player showed real cards this game (resets with State).
    vouched: Vec<u8>,
    opps: Vec<u8>,
    calls: Vec<u8>,
    prev_turn: Option<usize>,
    in_round: bool,
    /// A round has been logged in this game (deals before the first one can look like gains).
    seen_round: bool,
    /// The game redealt without a shot (an AFK player timed out); card gains are the deal.
    redealt: bool,
    /// Ranked matchmaking lobby (from the start event mode).
    ranked: bool,
    /// Devil mode seen (from the deck mode the plugin reports, or a Devil card showing up).
    devil_mode: bool,
    devil_active: bool,
    start_devil: bool,
    /// The current round's call was already reported; later pulls are extra Devil pulls.
    call_seen: bool,
    /// Plays this round, newest first, up to 4: (player, cards, true cards if mine, my Devil).
    recent: Vec<(usize, u8, u8, bool)>,
    /// The current call revealed the Devil; a Devil's Deal note was already printed.
    call_devil: bool,
    deal_noted: bool,
    /// A turn passed since the last pull (a later pull is a new call, not the same one).
    turn_since_pull: bool,
    /// Round claim total when I was last advised, so a turn is never advised twice.
    advised_at: Option<u8>,
    /// Disclosure suggestions already shown this round (player, their claim total).
    disclosed: Vec<(usize, u8)>,
    /// Think times per play this round: (seat, seconds). Cleared each round.
    round_thinks: Vec<(usize, f64)>,
    /// Times my bluff was exposed this game (a pull on me, or a Devil's Deal showing my fakes).
    /// The table learns a repeated opener, so each exposure raises the real-opening share (v57).
    my_caught: u8,
    /// This game only: times each player's play was called, and lies among them (`called`/`lies`
    /// are seeded from past-game profiles; the v61 data call model wants the in-game record).
    game_called: Vec<u8>,
    game_lies: Vec<u8>,
}

impl State {
    fn push_ev(&mut self, ev: crate::engine::Ev) {
        if self.evs.len() >= crate::engine::HIST_KEEP {
            self.evs.remove(0);
            self.round_ev = self.round_ev.and_then(|i| i.checked_sub(1));
        }
        self.evs.push(ev);
    }

    fn n(&self) -> usize {
        self.names.len()
    }

    fn name(&self, p: usize) -> String {
        let n = self.names.get(p).cloned().unwrap_or_else(|| format!("seat {p}"));
        if p == self.me { format!("{n} (you)") } else { n }
    }

    /// Grow every per-seat list to at least `n` seats (never shrink: seats can appear mid-game).
    fn ensure(&mut self, n: usize) {
        let n = n.max(self.names.len());
        for v in [&mut self.hands, &mut self.shots, &mut self.claims, &mut self.called, &mut self.lies, &mut self.opps, &mut self.calls] {
            v.resize(n, 0);
        }
        self.alive.resize(n, true);
        self.slots.resize(n, -1);
        self.looks.resize(n, 0);
        self.habit.resize(n, habit_of(0, 0));
        self.think_log.resize(n, 0.0);
        self.think_n.resize(n, 0);
        self.vouched.resize(n, 0);
        self.game_called.resize(n, 0);
        self.game_lies.resize(n, 0);
        self.skipped.resize(n, 0);
        self.calls_right.resize(n, 0);
        self.calls_judged.resize(n, 0);
        self.hab.resize(n, Default::default());
        self.luck_logp.resize(n, 0.0);
        self.flags.resize(n, Vec::new());

        self.pos.resize(n, (0.0, 0.0));
        while self.names.len() < n {
            self.names.push(format!("seat {}", self.names.len()));
        }
    }

    fn is_match(&self, c: i64) -> bool {
        c == self.table || c == JOKER
    }

    /// Player for a log seat index, if that index has been introduced by a seat line.
    fn player(&self, raw: i64) -> Option<usize> {
        if raw < 0 { None } else { self.raw.get(raw as usize).copied().flatten() }
    }

    fn my_match(&self) -> u8 {
        self.hand.iter().filter(|&&c| self.is_match(c)).count() as u8
    }

    fn my_devil(&self) -> bool {
        self.hand.contains(&DEVIL)
    }

    /// Seats in turn order: by the game's seat numbers when known, else by angle around the
    /// table centre. The direction comes from votes on observed turn hand-offs.
    fn turn_order(&self) -> Vec<usize> {
        let seats: Vec<usize> = (0..self.n()).collect();
        let mut by_slot = seats.clone();
        by_slot.sort_by_key(|&q| self.slots[q]);
        let distinct = by_slot.windows(2).all(|w| self.slots[w[0]] != self.slots[w[1]]);
        if !seats.is_empty() && seats.iter().all(|&q| self.slots[q] >= 0) && distinct {
            if self.dir_votes < 0 {
                by_slot.reverse();
            }
            return by_slot;
        }
        let n = seats.len();
        let (cx, cz) = seats.iter().fold((0.0, 0.0), |a, &q| (a.0 + self.pos[q].0 / n as f64, a.1 + self.pos[q].1 / n as f64));
        let have_pos = seats.iter().any(|&q| self.pos[q].0 != 0.0 || self.pos[q].1 != 0.0);
        let mut order = seats;
        if have_pos {
            order.sort_by(|&a, &b| {
                let ta = (self.pos[a].1 - cz).atan2(self.pos[a].0 - cx);
                let tb = (self.pos[b].1 - cz).atan2(self.pos[b].0 - cx);
                ta.partial_cmp(&tb).unwrap()
            });
            if self.dir_votes < 0 {
                order.reverse();
            }
        }
        order
    }

    fn vote_direction(&mut self, from: usize, to: usize) {
        let n = self.turn_order().len();
        if n < 3 {
            return;
        }
        let saved = self.dir_votes;
        self.dir_votes = 0;
        let order = self.turn_order();
        self.dir_votes = saved;
        let i = order.iter().position(|&s| s == from).unwrap();
        if order[(i + 1) % n] == to {
            self.dir_votes += 1;
        } else if order[(i + n - 1) % n] == to {
            self.dir_votes -= 1;
        }
    }

    /// The public state as an engine Game, with seats renumbered into turn order.
    /// Other players' hands are stored as card counts only.
    fn to_game(&self) -> (Game, usize, Vec<usize>) {
        let order = self.turn_order();
        let n = order.len();
        let mut rng = Rng::new(0);
        let mut alive = [false; MAX_PLAYERS];
        let mut shots = [0u8; MAX_PLAYERS];
        for (k, &s) in order.iter().enumerate() {
            // A player passed over twice is gone (left or AFK): not part of the game any more.
            alive[k] = self.alive[s] && self.skipped[s] < 2;
            shots[k] = self.shots[s].min(5);
        }
        let me = order.iter().position(|&s| s == self.me).unwrap();
        let mut g = Game::new_mode(n, alive, shots, me, self.devil_mode, &mut rng);
        g.devil_active = self.devil_active;
        g.ranked = self.ranked || std::env::var("LIARS_RANKED").map(|v| v == "1").unwrap_or(false);
        let mm = self.my_match();
        let md = self.my_devil() as u8;
        for (k, &s) in order.iter().enumerate() {
            if k == me {
                g.hm[k] = mm;
                g.hd[k] = md;
                g.hn[k] = self.hand.len() as u8 - mm - md;
                g.played_match[k] = self.start_match.saturating_sub(mm);
                g.played_devil[k] = self.start_devil && md == 0;
            } else {
                g.hm[k] = 0;
                g.hd[k] = 0;
                g.hn[k] = self.hands[s];
                g.played_match[k] = 0;
            }
            g.claims[k] = self.claims[s];
            g.rec_called[k] = self.called[s];
            g.rec_lies[k] = self.lies[s];
            g.rec_opps[k] = self.opps[s];
            g.rec_calls[k] = self.calls[s];
            g.rec_calls_right[k] = self.calls_right[s];
            g.prof[k] = self.hab.get(s).copied().unwrap_or_default();
        }
        g.claims_total = self.claims_total;
        // Plays made this round so far (the v61 call model reads a claim's position in the round).
        g.round_plays = self.round_plays.len().min(255) as u8;
        g.last = self.last.map(|(p, n)| Play { player: order.iter().position(|&s| s == p).unwrap(), n, n_true: 0, devil: false });
        // Plays for Devil's Deal, oldest first so the newest ends up in front.
        for &(p, cards, n_true, devil) in self.recent.iter().rev() {
            if let Some(k) = order.iter().position(|&s| s == p) {
                g.last = Some(Play { player: k, n: cards, n_true, devil });
                g.recent.copy_within(0..3, 1);
                g.recent[0] = g.last.unwrap();
                g.n_recent = (g.n_recent + 1).min(4);
            }
        }
        g.last = self.last.map(|(p, n)| Play { player: order.iter().position(|&s| s == p).unwrap(), n, n_true: 0, devil: false });
        let seat = |q: u8| order.iter().position(|&s| s == q as usize).unwrap_or(0) as u8;
        g.hist = self.evs.iter().map(|e| crate::engine::Ev { actor: seat(e.actor), target: seat(e.target), ..*e }).collect();
        g.current = me;
        (g, me, order)
    }

    /// Short warning list for a player, empty when nothing looks off.
    fn suspicion(&self, q: usize) -> String {
        let mut w: Vec<String> = self.flags[q].clone();
        let (right, judged) = (self.calls_right[q] as u32, self.calls_judged[q] as u32);
        // Real players are right on roughly half their calls; 6+ calls nearly all right is rare.
        if judged >= 6 && right * 10 >= judged * 9 {
            w.push(format!("calls right {right}/{judged}"));
        }
        let luck = self.luck_logp[q].exp();
        if luck < 0.02 {
            w.push(format!("revolver luck 1 in {:.0}", 1.0 / luck));
        }
        if w.is_empty() { String::new() } else { format!("  {RED}⚠ {}{RESET}", w.join(", ")) }
    }

    /// A player whose non-forced calls have been right 75%+ over 6+ calls (this game plus past
    /// games): real players land about 40%, so treat them as if they can see the cards.
    fn sharp_caller(&self, q: usize) -> Option<(u32, u32)> {
        let (right, judged) = (self.calls_right[q] as u32, self.calls_judged[q] as u32);
        if judged >= 6 && right * 4 >= judged * 3 { Some((right, judged)) } else { None }
    }

    fn only_one_with_cards(&self, p: usize) -> bool {
        (0..self.n()).all(|q| q == p || !self.alive[q] || self.hands[q] == 0)
    }
}

/// A player's record across past games: times called, caught lying, chances to call, calls,
/// calls that were right, and how many games it comes from.
#[derive(Default, Clone, Copy)]
pub struct Profile {
    called: u32,
    lies: u32,
    opps: u32,
    calls: u32,
    calls_right: u32,
    games: u32,
    /// Sum of ln(think seconds) over their plays, and how many (their usual pace).
    think_log: f64,
    think_n: u32,
    /// Card-check habit: their claims after someone else's (not opening a round), and how many
    /// of those they checked their cards before.
    look_n: u32,
    look_hits: u32,
    /// Habits the networks read (engine::Prof), counted like the engine does.
    hab: crate::engine::Prof,
}

/// What the network is told of a player's card-check habit (engine.rs Ev::habit): their
/// share of claims with a check, shrunk toward the lobby average 0.226, and claims seen.
fn habit_of(hits: u32, n: u32) -> (u8, u8) {
    let rate = (hits as f64 + 2.0 * 0.226) / (n as f64 + 2.0);
    ((rate * 255.0).round() as u8, n.min(30) as u8)
}

/// Think time relative to the player's usual pace (geometric mean of earlier plays, 4+ needed).
/// Held out by game, pace is a weak tell: 2x their usual time ~55% lies vs 50% (162 claims).
fn pace_label(think: f64, log_sum: f64, n: u32) -> Option<&'static str> {
    if n < 4 || think <= 0.0 {
        return None;
    }
    let r = think / (log_sum / n as f64).exp();
    Some(if r < 0.6 { "fast" } else if r < 1.4 { "normal" } else if r < 2.5 { "slow" } else { "very slow" })
}

// Lie detector v56: logistic regression refit on 6270 labelled opponent plays (spy + reveal), 2026-11-04.
// grouped-by-game CV AUC LR 0.655. class_weight=balanced, so 0.5 = population lie odds. Same 46 features as v51.
const LD_N: usize = 46;
const LD_MEAN: [f64; LD_N] = [
    1.5687400319, // n_claimed
    4.0531100478, // cards_before
    2.4843700159, // cards_left
    1.7879631203, // log_think
    3.9124401914, // index_in_round
    0.9473684211, // earlier_total
    0.6539074960, // n_earlier
    3.2995215311, // n_alive
    3.2154704944, // holders
    3.6354066986, // game_players
    4.7092503987, // round_num
    0.9593301435, // is_devil
    5.6679425837, // round_total
    3.1518341308, // total_claimed_by_others
    0.3571998177, // claim_ratio
    0.4326342371, // play_ratio
    0.8047619048, // round_ratio
    1.6600212653, // slot_mean
    2.0609250399, // slot_max
    1.2618819777, // slot_min
    0.7990430622, // slot_range
    0.1247208931, // emptying
    0.4326342371, // size_ratio
    0.2162679426, // is_opener
    0.4763955343, // is_late
    -0.2156923808, // think_vs_round
    -0.0262106872, // game_think_dev
    0.9942079717, // game_think_ratio
    0.1310137551, // profile_think_dev
    1.0481698776, // profile_think_ratio
    0.3856106881, // profile_lie_rate
    3.4496012759, // profile_n_labeled
    12.7671451356, // profile_n_plays
    0.5705935814, // cc_prob
    -1.2031942593, // cc_log
    0.2325358852, // cc_under_20
    1.0832535885, // prev_n
    1.2074449479, // prev_log_think
    0.4078149920, // prev_same_size
    0.5819605833, // round_before_ratio
    0.1360930736, // overshoot
    1.0125996810, // my_matching
    0.1446570973, // my_matching_ratio
    1.0331738437, // claimer_pulls
    0.2066347687, // claimer_pull_ratio
    1.2735247209, // my_pulls
];
const LD_SCALE: [f64; LD_N] = [
    0.7413373241, // n_claimed
    1.1762749270, // cards_before
    1.3008381706, // cards_left
    0.6729197965, // log_think
    2.6193650589, // index_in_round
    1.1781254255, // earlier_total
    0.7954194290, // n_earlier
    0.7046777589, // n_alive
    0.7368084287, // holders
    0.5435631327, // game_players
    3.3620682543, // round_num
    0.1975242244, // is_devil
    3.6180183598, // round_total
    2.8236146455, // total_claimed_by_others
    0.1845405207, // claim_ratio
    0.2553423487, // play_ratio
    0.5126609248, // round_ratio
    1.2836642349, // slot_mean
    1.4914282802, // slot_max
    1.3363993731, // slot_min
    1.2014607177, // slot_range
    0.3304021670, // emptying
    0.2553423487, // size_ratio
    0.4116990644, // is_opener
    0.4994425184, // is_late
    6.0184253440, // think_vs_round
    2.4823509016, // game_think_dev
    0.9603346605, // game_think_ratio
    2.9269469924, // profile_think_dev
    1.0360936996, // profile_think_ratio
    0.3146301509, // profile_lie_rate
    4.1534285525, // profile_n_labeled
    17.6991697870, // profile_n_plays
    0.3548984776, // cc_prob
    1.7879483561, // cc_log
    0.4224487511, // cc_under_20
    0.7771191254, // prev_n
    0.8770161648, // prev_log_think
    0.4914284529, // prev_same_size
    0.5201385422, // round_before_ratio
    0.2748039134, // overshoot
    1.0384939274, // my_matching
    0.1483562753, // my_matching_ratio
    1.2150767292, // claimer_pulls
    0.2430153458, // claimer_pull_ratio
    1.8030589418, // my_pulls
];
const LD_COEF: [f64; LD_N] = [
    0.2393162198, // n_claimed
    0.6521160262, // cards_before
    0.4532875022, // cards_left
    -0.0580947346, // log_think
    0.2466502661, // index_in_round
    1.0793507853, // earlier_total
    -0.1170600005, // n_earlier
    0.1259179411, // n_alive
    -0.0054235475, // holders
    -0.1140041501, // game_players
    0.0584393036, // round_num
    0.0665445945, // is_devil
    0.7139854222, // round_total
    0.4016793577, // total_claimed_by_others
    0.5450177444, // claim_ratio
    0.2119176774, // play_ratio
    -0.8013207894, // round_ratio
    -0.2748089572, // slot_mean
    -0.0354410979, // slot_max
    0.0618543226, // slot_min
    -0.1127959755, // slot_range
    -0.6038793595, // emptying
    0.2119176774, // size_ratio
    0.0282870786, // is_opener
    -0.1429102530, // is_late
    0.0943462398, // think_vs_round
    -0.0120716699, // game_think_dev
    0.0025364981, // game_think_ratio
    -0.0418828175, // profile_think_dev
    0.0102752722, // profile_think_ratio
    0.1084147042, // profile_lie_rate
    -0.0373619892, // profile_n_labeled
    0.0237484148, // profile_n_plays
    -0.0430401640, // cc_prob
    -0.1014132456, // cc_log
    -0.2011630311, // cc_under_20
    0.0043638810, // prev_n
    0.0528212155, // prev_log_think
    0.0304900126, // prev_same_size
    -1.2959405675, // round_before_ratio
    0.5901718080, // overshoot
    0.1196838890, // my_matching
    0.1196838890, // my_matching_ratio
    -0.0129775997, // claimer_pulls
    -0.0129775997, // claimer_pull_ratio
    0.0112597400, // my_pulls
];
const LD_INTERCEPT: f64 = -0.0355317307;

/// Lie detector: P(this claim is a lie) from game-time features.
/// `s` = game state, `claimer` = seat, `n_claimed` = cards they claimed,
/// `think` = their think time (seconds), `pt` = card-count P(all true),
/// `last_hi` = card-count P(this play true).
fn lie_detector(s: &State, claimer: usize, n_claimed: u8, think: f64, pt: f64, _last_hi: f64) -> f64 {
    let cards_left = s.hands[claimer];
    let cards_before = cards_left + n_claimed;
    let log_think = (think.max(0.5) as f64).ln();
    let index_in_round = s.round_plays.len() as f64;
    // Earlier claims by this player this round
    let earlier: Vec<u8> = s.round_plays.iter()
        .rev().skip(1) // skip current play
        .filter(|&&(p, _, _, _)| p == claimer)
        .map(|&(_, n, _, _)| n)
        .collect();
    let earlier_total: f64 = earlier.iter().map(|&n| n as f64).sum();
    let n_earlier = earlier.len() as f64;

    let n_alive = s.alive.iter().filter(|&&a| a).count() as f64;
    let holders = (0..s.names.len()).filter(|&q| s.alive[q] && s.hands[q] > 0).count() as f64;
    let game_players = s.names.len() as f64;
    let round_num = s.round_no as f64;
    let is_devil = if s.devil_mode { 1.0 } else { 0.0 };
    let round_total = s.claims_total as f64;
    let total_claimed_by_others: f64 = s.round_plays.iter()
        .rev().skip(1)
        .filter(|&&(p, _, _, _)| p != claimer)
        .map(|&(_, n, _, _)| n as f64)
        .sum();

    let truths: f64 = if s.devil_mode { 7.0 } else { 8.0 };
    let player_total = earlier_total + n_claimed as f64;
    let claim_ratio = player_total / truths.max(1.0);
    let play_ratio = n_claimed as f64 / (cards_before as f64).max(1.0);
    let round_ratio = round_total as f64 / truths.max(1.0);

    // Slot features: advisor doesn't track card positions, use training means
    let slot_mean = LD_MEAN[17];
    let slot_max = LD_MEAN[18];
    let slot_min = LD_MEAN[19];
    let slot_range = LD_MEAN[20];

    let emptying = if cards_left == 0 { 1.0 } else { 0.0 };
    let size_ratio = n_claimed as f64 / (cards_before as f64).max(1.0);
    let is_opener = if index_in_round <= 1.0 { 1.0 } else { 0.0 };
    let is_late = if index_in_round >= 4.0 { 1.0 } else { 0.0 };

    // Think time vs others this round (excluding current play)
    let other_thinks: Vec<f64> = s.round_thinks.iter()
        .rev().skip(1)
        .filter(|&&(_, t)| t > 0.0)
        .map(|&(_, t)| t)
        .collect();
    let think_vs_round = if other_thinks.is_empty() { 0.0 } else {
        think - other_thinks.iter().sum::<f64>() / other_thinks.len() as f64
    };

    // Within-game think deviation
    let prior_game: Vec<f64> = s.round_thinks.iter()
        .rev().skip(1)
        .filter(|&&(p, t)| p == claimer && t > 0.0)
        .map(|&(_, t)| t)
        .collect();
    let (game_think_dev, game_think_ratio) = if prior_game.is_empty() {
        (0.0, 1.0)
    } else {
        let mean = prior_game.iter().sum::<f64>() / prior_game.len() as f64;
        let std = if prior_game.len() > 1 {
            let var = prior_game.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / (prior_game.len() - 1) as f64;
            var.sqrt()
        } else {
            (mean * 0.5).max(1.0)
        };
        ((think - mean) / std.max(0.5), think / mean.max(0.5))
    };

    // Cross-game profile deviation (from the persistent think_log/think_n)
    let (profile_think_dev, profile_think_ratio) = if s.think_n[claimer] >= 2 {
        let geo_mean = (s.think_log[claimer] / s.think_n[claimer] as f64).exp();
        // Approximate std from log-normal: std ≈ geo_mean * 0.7
        let std_approx = (geo_mean * 0.7).max(1.0);
        ((think - geo_mean) / std_approx, think / geo_mean.max(0.5))
    } else {
        (0.0, 1.0)
    };

    // Profile lie rate from past games
    let profile_lie_rate = if s.called[claimer] > 0 {
        s.lies[claimer] as f64 / s.called[claimer] as f64
    } else {
        0.5 // prior
    };
    let profile_n_labeled = s.called[claimer] as f64;
    let profile_n_plays = s.think_n[claimer] as f64;

    // Card-count probability (hypergeometric, same value the advisor already computed)
    let cc_prob = pt.max(1e-6).min(1.0);
    let cc_log = cc_prob.ln();
    let cc_under_20 = if cc_prob < 0.20 { 1.0 } else { 0.0 };

    // Previous play context
    let prev = s.round_plays.iter().rev().skip(1).next(); // play before current
    let prev_n = prev.map(|&(_, n, _, _)| n as f64).unwrap_or(0.0);
    let prev_think = s.round_thinks.iter().rev().skip(1).next()
        .map(|&(_, t)| t).unwrap_or(0.0);
    let prev_log_think = if prev.is_some() { (prev_think.max(0.5)).ln() } else { 0.0 };
    let prev_same_size = if prev.is_some() && prev_n as u8 == n_claimed { 1.0 } else { 0.0 };
    let round_before = round_total - n_claimed as f64;
    let round_before_ratio = round_before / truths.max(1.0);
    let overshoot = ((round_total - truths).max(0.0)) / truths.max(1.0);

    // My hand info
    let my_matching = s.hand.iter().filter(|&&c| c == s.table as i64 || c == 4).count() as f64;
    let my_matching_ratio = my_matching / truths.max(1.0);

    // Pressure context
    let claimer_pulls = s.shots.get(claimer).copied().unwrap_or(0) as f64;
    let claimer_pull_ratio = claimer_pulls / 5.0;
    let my_pulls = s.shots.get(s.me).copied().unwrap_or(0) as f64;

    let feat = [
        n_claimed as f64, cards_before as f64, cards_left as f64, log_think,
        index_in_round, earlier_total, n_earlier, n_alive,
        holders, game_players, round_num, is_devil,
        round_total, total_claimed_by_others, claim_ratio, play_ratio,
        round_ratio, slot_mean, slot_max, slot_min,
        slot_range, emptying, size_ratio, is_opener,
        is_late, think_vs_round, game_think_dev, game_think_ratio,
        profile_think_dev, profile_think_ratio, profile_lie_rate, profile_n_labeled,
        profile_n_plays, cc_prob, cc_log, cc_under_20,
        prev_n, prev_log_think, prev_same_size, round_before_ratio, overshoot,
        my_matching, my_matching_ratio, claimer_pulls, claimer_pull_ratio, my_pulls,
    ];

    // Standard-scale then dot product
    let mut z = LD_INTERCEPT;
    for i in 0..LD_N {
        z += ((feat[i] - LD_MEAN[i]) / LD_SCALE[i]) * LD_COEF[i];
    }
    // Sigmoid
    1.0 / (1.0 + (-z).exp())
}

/// Builds per-name profiles from every deck-mode log in `dir` except `skip` (the live game).
pub fn build_profiles(dir: &Path, skip: Option<&Path>) -> HashMap<String, Profile> {
    let mut out: HashMap<String, Profile> = HashMap::new();
    let Ok(entries) = std::fs::read_dir(dir) else { return out };
    for path in entries.filter_map(|e| e.ok()).map(|e| e.path()) {
        let name_ok = path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("game_") && n.ends_with(".jsonl"));
        if !name_ok || skip == Some(path.as_path()) {
            continue;
        }
        let Ok(f) = File::open(&path) else { continue };
        let mut track = ProfTrack::default();
        let mut names: HashMap<i64, String> = HashMap::new();
        let mut seen: std::collections::HashSet<String> = Default::default();
        let mut table = 0i64;
        let mut last: Option<(i64, usize)> = None;
        let mut prev_last: Option<(i64, usize)> = None;
        let mut called_this_round = false;
        let mut chaos = false;
        let mut turn: i64 = -1;
        let mut looked: HashMap<i64, bool> = HashMap::new();
        for line in BufReader::new(f).lines().map_while(Result::ok) {
            if line.contains("\"e\":\"pose\"") || line.contains("\"e\":\"anchor\"") {
                continue;
            }
            let Ok(e) = serde_json::from_str::<Value>(&line) else { continue };
            track.feed(&e);
            let kind = e["e"].as_str().unwrap_or("");
            let raw = e["p"].as_i64().unwrap_or(-1);
            match kind {
                "start" => chaos = e["mode"].as_str() == Some("chaos"),
                "seat" => {
                    let n = e["name"].as_str().unwrap_or("?").to_string();
                    if e["me"].as_bool() != Some(true) {
                        seen.insert(n.clone());
                    }
                    names.insert(raw, n);
                }
                "turn" => turn = raw,
                "look" => {
                    if e["on"].as_bool() == Some(true) && raw != turn {
                        looked.insert(raw, true);
                    }
                }
                "round" => {
                    looked.clear();
                    table = e["table"].as_i64().unwrap_or(0);
                    last = None;
                    prev_last = None;
                    called_this_round = false;
                }
                "play" if !chaos => {
                    if last.is_some() {
                        if let Some(n) = names.get(&raw) {
                            let pr = out.entry(n.clone()).or_default();
                            pr.opps += 1;
                            pr.look_n += 1;
                            pr.look_hits += looked.get(&raw).copied().unwrap_or(false) as u32;
                        }
                    }
                    looked.remove(&raw);
                    let th = e["think"].as_f64().unwrap_or(-1.0);
                    if th > 0.3 && th < 40.0 {
                        if let Some(n) = names.get(&raw) {
                            let pr = out.entry(n.clone()).or_default();
                            pr.think_log += th.ln();
                            pr.think_n += 1;
                        }
                    }
                    prev_last = last;
                    last = Some((raw, e["n"].as_u64().unwrap_or(1) as usize));
                }
                "pull" if !chaos && !called_this_round => {
                    called_this_round = true;
                    let revealed = ints(&e["revealed"]);
                    if revealed.is_empty() || revealed.contains(&DEVIL) {
                        continue;
                    }
                    let lie = revealed.iter().any(|&c| c != table && c != JOKER);
                    let mut accused = e["accused"].as_i64().unwrap_or(-1);
                    let mut caller = e["caller"].as_i64().unwrap_or(-1);
                    // Same hand-clear correction as the live advisor.
                    if let (Some((lp, ln)), Some((pp, pn))) = (last, prev_last) {
                        let mismatch = revealed.len() != ln && revealed.len() == pn;
                        let puller = revealed.len() == pn && ((!lie && raw == lp) || (lie && raw == pp));
                        if lp != pp && (mismatch || puller) {
                            accused = pp;
                            caller = lp;
                        }
                    }
                    if let Some(n) = names.get(&accused) {
                        let pr = out.entry(n.clone()).or_default();
                        pr.called += 1;
                        pr.lies += lie as u32;
                    }
                    if let Some(n) = names.get(&caller) {
                        let pr = out.entry(n.clone()).or_default();
                        pr.opps += 1;
                        pr.calls += 1;
                        pr.calls_right += lie as u32;
                    }
                }
                _ => {}
            }
        }
        track.commit();
        for (n, h) in track.out {
            out.entry(n).or_default().hab.add(&h);
        }
        for n in seen {
            out.entry(n).or_default().games += 1;
        }
    }
    out
}

pub struct Watcher {
    /// Still reading the part of the log written before the advisor opened it: old turns are
    /// shown but not played out (the WHY section is too slow to run for every past turn).
    pub catching_up: bool,
    /// Neural network for Basic games (replaces the table bot's advice when loaded).
    pub net: Option<Arc<crate::nn::Model>>,
    /// Neural network for Devil games.
    pub net_devil: Option<Arc<crate::nn::Model>>,
    basic: Arc<Policy>,
    devil: Option<Arc<Policy>>,
    chaos_policy: Option<Arc<ChaosPolicy>>,
    /// Set while following a Chaos game; every event goes to it instead.
    chaos: Option<ChaosWatch>,
    st: State,
    rng: Rng,
    fallback: ProbAgent,
    /// Lines for another player's play, held until the next event: if that event is a call whose
    /// revealed cards don't match the play, it was the caller's hand being cleared, not a claim.
    held: RefCell<Option<String>>,
    /// The held play's size, and whether my turn advice is due after it.
    held_play: Option<(u8, bool)>,
    /// Records from past games, by player name.
    profiles: HashMap<String, Profile>,
    /// Habit counts by player name: past games, then this one as it is read.
    track: ProfTrack,
    /// Testing on real games (LIARS_EVAL_DUMP=file): my last call advice (claimer, call prob,
    /// card-count truth), labelled when I call that claim and the cards are shown.
    eval_pend: std::cell::Cell<Option<(usize, f64, f64, f64)>>,
    /// Plain networks for reading hands and playing out options (fast enough for thousands of games).
    pub read_net: Option<Arc<crate::nn::Model>>,
    pub read_net_devil: Option<Arc<crate::nn::Model>>,
    /// --auto: write the best action to aq_action.txt for the mod's auto-play.
    pub auto_play: bool,
    /// v61 data-fitted call model (LIARS_CALL_RULE=data; file from LIARS_CALL_DATA).
    pub data_call: Option<Arc<crate::datacall::DataCall>>,
    /// LIARS_CALL_RULE=ab: this game's arm (true = data call model, false = network call odds).
    pub ab_arm: Option<bool>,
}

impl Watcher {
    pub fn new(basic: Arc<Policy>, devil: Option<Arc<Policy>>, chaos_policy: Option<Arc<ChaosPolicy>>) -> Watcher {
        Watcher {
            net: None, net_devil: None, basic, devil, chaos_policy, chaos: None, st: State::default(), rng: Rng::new(0x5eed), fallback: ProbAgent::default(),
            held: RefCell::new(None), held_play: None, profiles: HashMap::new(), eval_pend: Default::default(), read_net: None, read_net_devil: None,
            catching_up: false,
            track: ProfTrack::default(),
            auto_play: false,
            data_call: None,
            ab_arm: None,
        }
    }

    fn handle(&mut self, e: &Value) {
        let kind = e["e"].as_str().unwrap_or("");
        if kind == "start" && e["mode"].as_str() == Some("chaos") {
            println!("\n=== new Chaos game log ===");
            if self.chaos_policy.is_none() {
                println!("(no chaos.bin found: Chaos advice uses a simple fallback)");
            }
            self.chaos = Some(ChaosWatch::new(self.chaos_policy.clone()));
            return;
        }
        if let Some(c) = self.chaos.as_mut() {
            c.handle(e);
            return;
        }
        // Side events (revolver chamber, card looks, aims, tamper flags) arrive mid-call; only
        // events that move the game on decide whether a held play was real.
        if matches!(kind, "turn" | "play" | "pull" | "round" | "seat") {
            self.flush_held(e);
        }
        self.track.feed(e);
        for q in 0..self.st.names.len().min(self.st.hab.len()) {
            self.st.hab[q] = self.track.out.get(&self.st.names[q]).copied().unwrap_or_default();
        }
        let raw = e["p"].as_i64().unwrap_or(-1);
        if kind == "seat" {
            self.on_seat(raw, e);
            return;
        }
        let p = match self.st.player(raw) {
            Some(q) => q,
            None if matches!(kind, "turn" | "play" | "pull" | "look") => return,
            None => 0,
        };
        match kind {
            "start" => {
                self.st = State::default();
                let mode = e["mode"].as_str().unwrap_or("?");
                self.st.ranked = mode == "deck_matchmaking";
                println!("\n=== new game log ({}{}) ===", mode, if self.st.ranked { " RANKED" } else { "" });
            }
            "round" => self.on_round(e),
            "turn" => self.on_turn(p, e),
            "play" => {
                if p == self.st.me {
                    self.eval_pend.set(None);
                }
                let other = p != self.st.me;
                if other {
                    *self.held.borrow_mut() = Some(String::new());
                }
                self.on_play(p, e);
                if other {
                    // The game sometimes skips my turn signal; if I'm next with cards, advise anyway.
                    let advise = self.st.in_round && self.next_holder(p) == Some(self.st.me);
                    self.held_play = Some((e["n"].as_u64().unwrap_or(1) as u8, advise));
                }
            }
            "pull" => self.on_pull(p, e),
            "hand" => {
                self.check_table(e);
                // My hand changed (the plugin reports it as soon as cards land). Before my first
                // play of the round, a full hand is my deal.
                let hand = ints(&e["hand"]);
                let s = &mut self.st;
                if s.in_round && s.claims.get(s.me).copied() == Some(0) && hand.len() == crate::engine::HAND_SIZE as usize && hand != s.start_hand {
                    s.start_hand = hand.clone();
                    s.hand = hand;
                    s.start_match = s.my_match();
                    s.start_devil = s.my_devil();
                    let cards: Vec<&str> = s.hand.iter().map(|&c| card_name(c)).collect();
                    println!("  {DIM}your hand: {} ({} matching){RESET}", cards.join(" "), s.start_match);
                } else if s.in_round && !hand.is_empty() {
                    s.hand = hand;
                }
            }
            "dealcards" => self.on_deal_cards(e),
            "deal" => {
                let name = e["caller"].as_str().unwrap_or("");
                let s = &mut self.st;
                s.pending_deal = (0..s.n()).find(|&q| s.names[q] == name);
            }
            "redeal" => {
                self.st.redealt = true;
                println!("  {YELLOW}cards redealt with no shot (a player probably timed out){RESET}");
            }
            "tamper" => {
                let what = e["what"].as_str().unwrap_or("?").to_string();
                if what == "gained cards mid-round" && self.st.redealt {
                    return;
                }
                // The plugin used to carry an unfinished round over from the previous game, so the
                // new game's deal read as gaining cards before any round was logged.
                if what == "gained cards mid-round" && !self.st.seen_round {
                    return;
                }
                // Never flag myself, and ignore card-count flags in a log's first round: a round
                // logged before the real deal makes the deal itself look like cards appearing.
                if p == self.st.me || (self.st.round_no <= 1 && what != "revolver changed outside a pull") {
                    return;
                }
                println!("  {RED}{BOLD}⚠ CHEATING? {}{RESET}: {what}", self.st.name(p));
                if !self.st.flags[p].contains(&what) {
                    self.st.flags[p].push(what);
                }
            }
            "look" => {
                // On a player's own turn the game opens their cards; only checks while someone
                // else is playing are a choice.
                if e["on"].as_bool() == Some(true) && p < self.st.n() && self.st.prev_turn != Some(p) {
                    self.st.looks[p] = self.st.looks[p].saturating_add(1);
                }
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
        let s = &mut self.st;
        let known = (0..s.n()).find(|&q| s.names[q] == name && s.slots[q] == slot);
        let q = known.unwrap_or_else(|| {
            let q = s.n();
            s.ensure(q + 1);
            s.names[q] = name.clone();
            s.slots[q] = slot;
            q
        });
        let r = raw as usize;
        if s.raw.len() <= r {
            s.raw.resize(r + 1, None);
        }
        s.raw[r] = Some(q);
        let pos = ints_f(&e["pos"]);
        if pos.len() == 3 {
            s.pos[q] = (pos[0], pos[2]);
        }
        let me = e["me"].as_bool() == Some(true);
        if me {
            s.me = q;
        }
        if known.is_none() {
            let mut note = String::new();
            if !me {
                if let Some(pr) = self.profiles.get(&name).copied() {
                    let s = &mut self.st;
                    let c = |v: u32| v.min(255) as u8;
                    s.called[q] = c(pr.called);
                    s.lies[q] = c(pr.lies);
                    s.opps[q] = c(pr.opps);
                    s.calls[q] = c(pr.calls);
                    s.calls_right[q] = c(pr.calls_right);
                    s.calls_judged[q] = c(pr.calls); // same non-forced calls the right-count comes from
                    s.think_log[q] = pr.think_log;
                    s.think_n[q] = pr.think_n;
                    s.habit[q] = habit_of(pr.look_hits, pr.look_n);
                    let h = pr.hab;
                    note = format!("  {YELLOW}known from {} past game{}: caught lying {}/{}, calls {}/{} (right {}), fast dumps {}/{} rounds, hand-emptying claims fake {}/{} when called, 1v1 lies {}/{}{RESET}",
                        pr.games, if pr.games == 1 { "" } else { "s" }, pr.lies, pr.called, pr.calls, pr.opps, pr.calls_right,
                        h.dumps, h.rounds, h.empty_lies, h.empty_called, h.duel_lies, h.duel_called);
                }
            }
            println!("seat {q}: {}{}{note}", name, if me { " (you)" } else { "" });
        }
    }

    fn on_round(&mut self, e: &Value) {
        let s = &mut self.st;
        let hands = ints(&e["hands"]);
        let alive_now = ints(&e["alive"]).iter().filter(|&&v| v != 0).count() as u8;
        s.push_ev(crate::engine::Ev { kind: 5, n: alive_now, ..Default::default() });
        s.round_no += 1;
        s.round_ev = Some(s.evs.len() - 1);
        s.table = e["table"].as_i64().unwrap_or(0);
        s.hand = ints(&e["hand"]);
        // A fresh deal is always a full hand; fewer cards are last round's leftovers, logged
        // before the new cards arrived. Wait for my turn to bring the real hand instead.
        if s.hand.len() != crate::engine::HAND_SIZE as usize {
            s.hand.clear();
        }
        s.start_match = s.my_match();
        s.start_devil = s.my_devil();
        s.start_hand = s.hand.clone();
        // The deck report lists many Devil-related fields in every mode; only DeckMode decides.
        // (Older Blorf modes report the bare mode name.)
        if let Some(d) = e["deck"].as_str() {
            let mode = match d.find("DeckMode=") {
                Some(i) => d[i + 9..].split(';').next().unwrap_or(""),
                None => d,
            };
            if mode != "?" && !mode.is_empty() {
                s.devil_mode = mode.eq_ignore_ascii_case("devil");
            }
        }
        if s.start_devil {
            s.devil_mode = true;
        }
        for (i, &h) in hands.iter().enumerate() {
            if let Some(q) = s.player(i as i64) {
                s.hands[q] = h as u8;
            }
        }
        for (i, &v) in ints(&e["pulls"]).iter().enumerate() {
            if let Some(q) = s.player(i as i64) {
                s.shots[q] = v as u8;
            }
        }
        // Death is final within a game: the game can reset a dead (often departed) player's seat
        // to "alive" with 0 pulls. Only a genuinely new game revives: the first round of a log, with
        // everyone alive and at 0 pulls. (Someone dying on their first pull while everyone else is
        // also at 0 looks the same otherwise.)
        let new_game = e["n"].as_i64() == Some(1)
            && ints(&e["pulls"]).iter().all(|&v| v == 0)
            && ints(&e["alive"]).iter().all(|&v| v != 0);
        for (i, &v) in ints(&e["alive"]).iter().enumerate() {
            if let Some(q) = s.player(i as i64) {
                s.alive[q] = v != 0 && (new_game || s.alive[q]);
            }
        }
        // Opponents' cards can still be mid-deal when the round is logged; everyone alive gets 5.
        for q in 0..s.n() {
            if s.alive[q] && s.hands[q] == 0 {
                s.hands[q] = crate::engine::HAND_SIZE;
            }
        }
        s.claims.iter_mut().for_each(|c| *c = 0);
        s.prev_last = None;
        s.recent.clear();
        s.round_plays.clear();
        s.round_thinks.clear();
        s.pending_deal = None;
        s.disclosed.clear();
        s.claims_total = 0;
        s.last = None;
        s.prev_turn = None;
        s.in_round = true;
        s.seen_round = true;
        s.redealt = false;
        s.call_seen = false;
        s.advised_at = None;
        s.devil_active = s.devil_mode && (0..s.n()).filter(|&q| s.alive[q]).count() >= 3;
        let hand: Vec<&str> = s.hand.iter().map(|&c| card_name(c)).collect();
        // The round can be logged before the deal animation finishes; the hand shows on my turn.
        let hand_txt = if hand.is_empty() { "(still being dealt)".to_string() } else { format!("{} ({} matching)", hand.join(" "), s.start_match) };
        println!("\n--- round {} | table card: {} | your hand: {}{} ---",
            e["n"], card_name(s.table), hand_txt,
            if s.devil_active { if s.start_devil { " | you hold the DEVIL" } else { " | Devil in play" } } else { "" });
        for q in 0..s.n() {
            if s.alive[q] {
                let rec = if s.called[q] > 0 { format!(", caught lying {}/{}", s.lies[q], s.called[q]) } else { String::new() };
                let cr = if s.opps[q] > 0 { format!(", calls {}/{}", s.calls[q], s.opps[q]) } else { String::new() };
                println!("    {:<20} pulls {}/6{}{}{}", s.name(q), s.shots[q], rec, cr, s.suspicion(q));
            }
        }
    }

    fn on_play(&mut self, p: usize, e: &Value) {
        let passed_on = self.st.last.map(|(c, _)| c).filter(|&c| c != p);
        let hand_before = self.st.hands.get(p).copied().unwrap_or(0);
        let s = &mut self.st;
        if !s.in_round || p >= s.n() {
            return;
        }
        let n = e["n"].as_u64().unwrap_or(1) as u8;
        if s.last.is_some() {
            s.opps[p] = s.opps[p].saturating_add(1);
        }
        s.claims[p] += n;
        s.claims_total += n;
        if let Some(i) = s.round_ev.take() {
            if s.claims_total == n {
                s.evs[i].actor = p as u8;
            }
        }
        let left = e["left"].as_u64().unwrap_or(0) as u8;
        let total = s.claims_total;
        // Tells for the network: card checks while waiting and think time against their usual
        // pace (same units as the simulated ones in engine.rs `tells`).
        let th = e["think"].as_f64().unwrap_or(-1.0);
        let pace = if th > 0.0 && s.think_n[p] >= 4 {
            ((th.max(0.3) / (s.think_log[p] / s.think_n[p] as f64).exp()).log2().clamp(-3.9, 3.9) * 32.0).round() as i8
        } else {
            0
        };
        s.push_ev(crate::engine::Ev { kind: 1, actor: p as u8, n, left, total, looks: s.looks[p].min(6), pace, habit: s.habit[p].0, habit_n: s.habit[p].1, ..Default::default() });
        s.skipped[p] = 0;
        s.prev_last = s.last;
        s.last = Some((p, n));
        s.hands[p] = e["left"].as_u64().unwrap_or(0) as u8;
        let (mut n_true, mut my_devil) = (0, false);
        if p == s.me && e.get("hand").is_some() {
            let (mm, md) = (s.my_match(), s.my_devil());
            s.hand = ints(&e["hand"]);
            n_true = mm.saturating_sub(s.my_match()).min(n);
            my_devil = md && !s.my_devil();
        }
        s.recent.insert(0, (p, n, n_true, my_devil));
        s.round_plays.push((p, n, n_true, my_devil));
        s.recent.truncate(4);
        let think = e["think"].as_f64().unwrap_or(-1.0);
        if think > 0.0 { s.round_thinks.push((p, think)); }
        let pace = if p != s.me { pace_label(think, s.think_log[p], s.think_n[p]) } else { None };
        if think > 0.3 && think < 40.0 {
            s.think_log[p] += think.ln();
            s.think_n[p] += 1;
        }
        s.last_pace = pace;
        let pace_txt = match pace {
            Some("fast") => format!(" {RESET}{GREEN}FAST for them{RESET}{DIM}"),
            Some("very slow") => format!(" {RESET}{YELLOW}VERY SLOW for them{RESET}{DIM}"),
            Some(l) => format!(" {l} for them"),
            None => String::new(),
        };
        let looks = if s.looks[p] > 0 { format!(", checked cards {}x while waiting", s.looks[p]) } else { String::new() };
        s.looks[p] = 0;
        let line = format!("  {BOLD}{}{RESET} claims {} {}  {DIM}({} left, {:.1}s{pace_txt}{looks}, round total {}){RESET}",
            s.name(p), n, card_name(s.table), s.hands[p], think, s.claims_total);
        // My cards this round are known even after I've played them all.
        let show_odds = p != s.me && !s.start_hand.is_empty();
        self.out(line);
        if show_odds {
            self.show_claim_odds(n);
        }
        if let Some(c) = passed_on {
            self.show_pass_inference(p, c, hand_before);
        }
        if p != self.st.me {
            self.show_next_threshold(p);
        }
        self.show_disclosure();
    }

    /// Prints now, or appends to the held-back claim lines while a play is pending.
    fn out(&self, line: String) {
        match self.held.borrow_mut().as_mut() {
            Some(buf) => {
                buf.push_str(&line);
                buf.push('\n');
            }
            None => println!("{line}"),
        }
    }

    /// Releases (or, for a hand-clear, drops) the held play before handling the next event.
    fn flush_held(&mut self, e: &Value) {
        let Some((n, advise)) = self.held_play.take() else { return };
        let text = self.held.borrow_mut().take().unwrap_or_default();
        let revealed = ints(&e["revealed"]);
        if e["e"] == "pull" && !revealed.is_empty() && revealed.len() != n as usize {
            return; // the "play" was the caller's hand being cleared
        }
        print!("{text}");
        if advise {
            self.advise_once();
        }
    }

    /// True cards in the deck this round (table rank + Jokers, minus the Devil when in play).
    fn truths(&self) -> u8 {
        MATCHING_IN_DECK - self.st.devil_active as u8
    }

    /// The fewest real cards someone would need to hold to know `c`'s claims can't all be true.
    fn proof_threshold(&self, c: usize) -> Option<u8> {
        let k = self.truths() as i32 - self.st.claims[c] as i32 + 1;
        (k >= 1).then_some(k as u8)
    }

    /// Under a claim: the next player could prove it false with this many real cards.
    fn show_next_threshold(&self, p: usize) {
        let s = &self.st;
        let Some(next) = self.next_holder(p) else { return };
        if next == s.me {
            return;
        }
        if let Some(k) = self.proof_threshold(p) {
            if k <= s.hands[next] {
                // Their total can be proven false by the next player's own hand, but only if they
                // really hold that many: say how likely that is, and stay quiet when it isn't.
                let unseen = self.truths().saturating_sub(s.start_match) as u64;
                let pop = (DECK_SIZE - HAND_SIZE) as u64;
                let pr = hypergeom_at_least(k as u64, pop, unseen, s.hands[next] as u64);
                if pr >= 0.15 {
                    self.out(format!("     {DIM}{} can tell {} lied this round if they hold {k}+ real {} (~{:.0}% chance){RESET}",
                        s.name(next), s.name(p), card_name(s.table), pr * 100.0));
                }
            }
        }
    }

    /// When a player plays instead of calling, they likely didn't have the cards to prove it.
    fn show_pass_inference(&self, p: usize, c: usize, hand_before: u8) {
        let s = &self.st;
        if p == s.me || c == s.me {
            return;
        }
        if let Some(k) = self.proof_threshold(c) {
            if k <= hand_before {
                self.out(format!("     {DIM}{} passed on {}'s claim -> probably holds at most {} real {}{RESET}",
                    s.name(p), s.name(c), k - 1, card_name(s.table)));
            }
        }
    }

    /// When my hand can't hurt me any more (empty, or only real cards left), saying how many real
    /// cards I had can expose someone's claims for the whole table.
    fn show_disclosure(&mut self) {
        let me = self.st.me;
        let all_real = !self.st.hand.is_empty() && self.st.hand.iter().all(|&c| self.st.is_match(c));
        if !(self.st.hand.is_empty() || all_real) || self.st.start_match == 0 {
            return;
        }
        let r = self.st.start_match;
        let left_for_others = self.truths().saturating_sub(r);
        for q in 0..self.st.n() {
            let cq = self.st.claims[q];
            if q == me || !self.st.alive[q] || cq == 0 || cq <= left_for_others || self.st.disclosed.contains(&(q, cq)) {
                continue;
            }
            self.st.disclosed.push((q, cq));
            let why = if self.st.hand.is_empty() { "you're out of cards, safe this round" } else { "your cards are all real anyway" };
            self.out(format!("  {YELLOW}{BOLD}worth saying:{RESET} \"I had {r} real {}\"  {DIM}({why}){RESET}", card_name(self.st.table)));
            self.out(format!("     {DIM}-> then {}'s {cq} claimed {} can't all be true (only {left_for_others} left for everyone else){RESET}",
                self.st.name(q), card_name(self.st.table)));
        }
    }

    /// The 4 cards a Devil's Deal showed are the last 4 thrown this round, in throw order (game
    /// code: TakeLast(4)), so each belongs to a known play: credit every exposed play to its
    /// player's record like a called one, and give the network the reveal.
    fn on_deal_cards(&mut self, e: &Value) {
        let cards = ints(&e["round_cards"]);
        let s = &mut self.st;
        if cards.len() < 4 {
            return;
        }
        let shown = &cards[cards.len() - 4..];
        let table = s.table;
        let fake = |c: i64| c != table && c != JOKER && c != DEVIL;
        // Plays newest first; a trailing play by the deal caller is their hand being cleared.
        let mut plays: Vec<(usize, u8)> = s.round_plays.iter().map(|&(p, n, _, _)| (p, n)).collect();
        if let (Some(&(p, _)), Some(dc)) = (plays.last(), s.pending_deal) {
            if p == dc {
                plays.pop();
            }
        }
        let total = s.claims_total;
        let mut end = 4usize;
        let mut parts = Vec::new();
        for &(p, n) in plays.iter().rev() {
            if end == 0 {
                break;
            }
            let k = (n as usize).min(end);
            let mine = &shown[end - k..end];
            end -= k;
            let fakes = mine.iter().filter(|&&c| fake(c)).count() as u8;
            let full = k == n as usize;
            if p < s.n() && (full || fakes > 0) {
                s.called[p] = s.called[p].saturating_add(1);
                s.game_called[p] = s.game_called[p].saturating_add(1);
                if fakes > 0 {
                    s.lies[p] = s.lies[p].saturating_add(1);
                    s.game_lies[p] = s.game_lies[p].saturating_add(1);
                    if p == s.me {
                        s.my_caught = s.my_caught.saturating_add(1);
                    }
                }
            }
            s.push_ev(crate::engine::Ev { kind: 6, actor: p as u8, n: k as u8, left: full as u8, total, flag: fakes, ..Default::default() });
            let what = if fakes == 0 { "real".to_string() } else if fakes as usize == k { "FAKE".to_string() } else { format!("{fakes} fake") };
            let of = if full { String::new() } else { format!(" of their {n}") };
            parts.push(format!("{}'s {k}{of}: {what}", s.name(p)));
        }
        let names: Vec<&str> = shown.iter().map(|&c| card_name(c)).collect();
        println!("  {YELLOW}Devil's Deal showed [{}]{RESET}  ->  {}", names.join(" "), parts.join(" | "));
    }

    /// LIARS_SCOUT=name: when that player faces a claim, what our network would do in their
    /// seat. Their hand is hidden, so the call chance is averaged over every count of real cards
    /// they could hold (weighted by the card count from public plays alone).
    fn scout(&self, p: usize) {
        let Ok(target) = std::env::var("LIARS_SCOUT") else { return };
        if self.st.names.get(p) != Some(&target) || self.st.last.is_none() || p == self.st.me {
            return;
        }
        let devil = self.st.devil_mode;
        let Some(net) = (if devil { self.net_devil.clone() } else { self.net.clone() }) else { return };
        let size = self.st.hands.get(p).copied().unwrap_or(0) as usize;
        if size == 0 {
            return;
        }
        let other = if self.st.table == 1 { 2 } else { 1 };
        let truths = self.truths() as u64;
        let pop = (DECK_SIZE - HAND_SIZE) as u64 + HAND_SIZE as u64;
        let (mut wsum, mut call) = (0.0, 0.0);
        let mut by_k = Vec::new();
        for k in 0..=size.min(truths as usize) {
            // Chance of holding exactly k real cards among `size` (public view, deck of 20).
            let w = hypergeom_at_least(k as u64, pop, truths, size as u64) - hypergeom_at_least(k as u64 + 1, pop, truths, size as u64);
            if w <= 0.0 {
                continue;
            }
            let mut st = self.st.clone();
            st.me = p;
            st.hand = (0..size).map(|i| if i < k { self.st.table } else { other }).collect();
            st.start_match = k as u8;
            st.start_devil = false;
            let (g, me, _) = st.to_game();
            let pc = crate::nn::Net::probs(&net.logits(&g, me, crate::nn::KIND_CALL), 2, 2)[1] as f64;
            wsum += w;
            call += w * pc;
            by_k.push(format!("{k}:{:.0}%", pc * 100.0));
        }
        if wsum > 0.0 {
            let (c, n) = self.st.last.unwrap();
            println!("SCOUT {} faces {}'s {} -> our network calls {:.0}% (by real cards held {})",
                target, self.st.name(c), n, call / wsum * 100.0, by_k.join(" "));
        }
    }

    /// One compact line under another player's claim: how likely it is to be true from what my
    /// hand shows (all their claims this round, and just this play if earlier ones were bluffs).
    fn show_claim_odds(&self, n: u8) {
        let (g, me, _) = self.st.to_game();
        let all = p_claim_true(&g, me);
        let unseen = g.truths().saturating_sub(g.hm[me] + g.played_match[me]) as u64;
        let this = hypergeom_at_least(n as u64, (DECK_SIZE - HAND_SIZE) as u64, unseen, HAND_SIZE as u64).max(all);
        self.out(format!("     {DIM}card count: all their claims true {:.0}%  |  this play true: {:.0}-{:.0}%{RESET}",
            all * 100.0, all * 100.0, this * 100.0));
        let (_, _, order) = self.st.to_game();
        let name = |k: usize| self.st.name(order[k]);
        for line in proof_lines(&g, me, n, this, card_name(self.st.table), &name) {
            self.out(format!("     {line}"));
        }
    }

    fn on_pull(&mut self, p: usize, e: &Value) {
        let s = &mut self.st;
        if p >= s.n() {
            return;
        }
        let died = e["died"].as_bool().unwrap_or(false);
        if !died {
            let k = s.shots[p].min(4) as f64;
            s.luck_logp[p] += ((5.0 - k) / (6.0 - k)).ln();
        }
        let new_call = s.call_seen && s.turn_since_pull;
        s.turn_since_pull = false;
        if new_call {
            // Another call with no round seen in between: the plugin stops seeing deals and plays
            // once I'm dead, and its caller/accused are stale then. Show only who pulled.
            s.shots[p] = e["pulls"].as_u64().unwrap_or(s.shots[p] as u64) as u8;
            if died {
                s.alive[p] = false;
            }
            println!("  {DIM}(spectating: plays not visible){RESET} {} pulls ({}/6){}", s.name(p), s.shots[p],
                if died { " and DIES" } else { ", survives" });
            return;
        }
        if s.call_seen {
            // Further pulls from the same call: a revealed Devil makes several players shoot.
            s.shots[p] = e["pulls"].as_u64().unwrap_or(s.shots[p] as u64) as u8;
            if died {
                s.alive[p] = false;
            }
            // Several shooters without a Devil on the table: that was a Devil's Deal.
            if !s.call_devil && !s.deal_noted {
                s.deal_noted = true;
                println!("    {MAGENTA}{BOLD}^ that was a DEVIL'S DEAL{RESET}{MAGENTA}, not a Liar call: the last 4 cards were all lies, so everyone else still holding cards shoots{RESET}");
            }
            s.push_ev(crate::engine::Ev { kind: 4, actor: p as u8, left: s.shots[p], flag: died as u8, ..Default::default() });
            println!("    {} also pulls ({}/6){}", s.name(p), s.shots[p], if died { " and DIES" } else { ", survives" });
            return;
        }
        s.call_seen = true;
        // A Devil's Deal: if the caller pulls alone, it failed (one of the last 4 cards was real);
        // otherwise everyone else with cards is pulling. Not a Liar call on the last player.
        if let Some(dc) = s.pending_deal.take() {
            s.deal_noted = true;
            s.call_devil = false;
            s.shots[p] = e["pulls"].as_u64().unwrap_or(s.shots[p] as u64) as u8;
            if died {
                s.alive[p] = false;
            }
            s.in_round = false;
            if dc < s.n() && !s.only_one_with_cards(dc) {
                s.opps[dc] = s.opps[dc].saturating_add(1);
                s.calls[dc] = s.calls[dc].saturating_add(1);
            }
            let target = s.last.map_or(0, |(q, _)| q as u8);
            let total = s.claims_total;
            s.push_ev(crate::engine::Ev { kind: 3, actor: dc as u8, target, n: 4, total, flag: (p != dc) as u8, ..Default::default() });
            s.push_ev(crate::engine::Ev { kind: 4, actor: p as u8, left: s.shots[p], flag: died as u8, ..Default::default() });
            let tail = format!("{} pulls ({}/6){}", s.name(p), s.shots[p], if died { " and DIES" } else { ", survives" });
            if p == dc {
                println!("  {MAGENTA}{BOLD}DEVIL'S DEAL{RESET}{MAGENTA} by {} FAILED{RESET} (one of the last 4 cards was real) -> {tail}", s.name(dc));
            } else {
                println!("  {MAGENTA}{BOLD}DEVIL'S DEAL{RESET}{MAGENTA} by {} WON{RESET} (the last 4 were all lies: everyone else with cards shoots) -> {tail}", s.name(dc));
            }
            return;
        }
        let revealed = ints(&e["revealed"]);
        let devil = revealed.contains(&DEVIL);
        s.call_devil = devil;
        s.deal_noted = false;
        if devil {
            s.devil_mode = true;
        }
        let mut accused = s.player(e["accused"].as_i64().unwrap_or(-1)).map_or(-1, |q| q as i64);
        let mut caller = s.player(e["caller"].as_i64().unwrap_or(-1)).map_or(-1, |q| q as i64);
        // The revealed cards must be the size of the play that was called. If they don't match the
        // last "play", that was the forced caller's hand being cleared: the real accused made the
        // play before it, and the "player" is the caller.
        // Lie or truth comes from the revealed cards themselves (table rank and Jokers are true).
        let lie = !devil && !revealed.is_empty() && revealed.iter().any(|&c| c != s.table && c != JOKER);
        if let (Some((lp, ln)), Some((pp, pn))) = (s.last, s.prev_last) {
            // The last "play" may really be the caller's hand being cleared as they called. Tell-tale
            // signs: the revealed cards don't match its size, or the wrong person pulled (on a true
            // reveal the caller pulls, on a lie the liar pulls).
            let size_mismatch = !revealed.is_empty() && revealed.len() != ln as usize && revealed.len() == pn as usize;
            let puller_says_clear = !devil && !revealed.is_empty() && revealed.len() == pn as usize
                && ((!lie && p == lp) || (lie && p == pp));
            if lp != pp && (size_mismatch || puller_says_clear) {
                s.claims[lp] = s.claims[lp].saturating_sub(ln);
                s.claims_total = s.claims_total.saturating_sub(ln);
                if !s.recent.is_empty() {
                    s.recent.remove(0);
                }
                accused = pp as i64;
                caller = lp as i64;
            }
        }
        if let Some((cl, pc, pt, rl)) = self.eval_pend.take() {
            if caller == s.me as i64 && accused == cl as i64 {
                if let Ok(path) = std::env::var("LIARS_EVAL_DUMP") {
                    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                        use std::io::Write as _;
                        let _ = writeln!(f, "{},{:.5},{:.5},{},{},{:.5}", s.name(cl).replace(',', " "), pc, pt, lie as u8, devil as u8, rl);
                    }
                }
            }
        }
        // Testing call rules on every called claim, whoever called it (LIARS_ALL_DUMP=file): the card
        // count from my seat at the moment of the call, and what the cards turned out to be.
        if let Ok(path) = std::env::var("LIARS_ALL_DUMP") {
            let st = &self.st;
            if accused >= 0 && accused != st.me as i64 && !st.hand.is_empty() && st.alive.get(st.me).copied().unwrap_or(false) {
                let (g, me, order) = st.to_game();
                if let Some(lp) = g.last {
                    if order[lp.player] == accused as usize {
                        let pt = p_claim_true(&g, me);
                        let unseen = g.truths().saturating_sub(g.hm[me] + g.played_match[me]) as u64;
                        let last_hi = hypergeom_at_least(lp.n as u64, (DECK_SIZE - HAND_SIZE) as u64, unseen, HAND_SIZE as u64);
                        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                            use std::io::Write as _;
                            let _ = writeln!(f, "{},{:.5},{:.5},{},{},{},{},{},{},{},{},{}", st.name(accused as usize).replace(',', " "),
                                pt, last_hi.max(pt), lp.n, g.hand(lp.player), g.n_alive(), lie as u8, devil as u8,
                                (caller == st.me as i64) as u8, g.hm[me], st.calls.get(accused as usize).copied().unwrap_or(0),
                                g.is_forced(me) as u8);
                        }
                    }
                }
            }
        }
        let s = &mut self.st;
        if accused >= 0 && (accused as usize) < s.n() {
            let a = accused as usize;
            s.called[a] = s.called[a].saturating_add(1);
            s.game_called[a] = s.game_called[a].saturating_add(1);
            if lie {
                s.lies[a] = s.lies[a].saturating_add(1);
                s.game_lies[a] = s.game_lies[a].saturating_add(1);
                if a == s.me {
                    s.my_caught = s.my_caught.saturating_add(1);
                }
            }
        }
        if caller >= 0 && (caller as usize) < s.n() {
            let c = caller as usize;
            if !s.only_one_with_cards(c) {
                s.opps[c] = s.opps[c].saturating_add(1);
                s.calls[c] = s.calls[c].saturating_add(1);
                s.calls_judged[c] = s.calls_judged[c].saturating_add(1);
                if lie {
                    s.calls_right[c] = s.calls_right[c].saturating_add(1);
                }
            }
        }
        // My call turned up real cards: this player has shown me the goods once. Make the lie
        // detector need more before I test them again this game (2026-11-03: two wrong calls on
        // the same player's real triples at 73% and 76% in the first ranked game under v51).
        if caller >= 0 && caller as usize == s.me && !devil && accused >= 0 && (accused as usize) < s.n() {
            let a = accused as usize;
            if a < s.vouched.len() {
                s.vouched[a] = if lie { 0 } else { s.vouched[a].saturating_add(1) };
            }
        }
        // Anyone caught lying by anyone has shown they bluff: drop the benefit of the doubt.
        if lie && !devil && accused >= 0 && (accused as usize) < s.vouched.len() {
            s.vouched[accused as usize] = 0;
        }
        // Impossible reveal: the revealed cards plus my own hand need more copies than exist.
        if accused >= 0 && accused as usize != s.me && !s.start_hand.is_empty() {
            for code in [1i64, 2, 3, 4, DEVIL] {
                let mut total = (if code == 4 { 2 } else if code == DEVIL { 1 } else { 6 }) as usize;
                if code == s.table && s.devil_active {
                    total -= 1; // one table-rank card is the Devil this round
                }
                let shown = revealed.iter().filter(|&&c| c == code).count();
                let mine = s.start_hand.iter().filter(|&&c| c == code).count();
                if shown > 0 && shown + mine > total {
                    let a = accused as usize;
                    let f = format!("revealed {shown}x {} while you held {mine} (only {total} exist)", card_name(code));
                    println!("  {RED}{BOLD}⚠ CHEATING? {}{RESET}: {f}", s.name(a));
                    s.flags[a].push("impossible cards".into());
                }
            }
        }
        s.shots[p] = e["pulls"].as_u64().unwrap_or(s.shots[p] as u64) as u8;
        if died {
            s.alive[p] = false;
        }
        s.in_round = false;
        if caller >= 0 && accused >= 0 {
            let flag = if devil { 2 } else { lie as u8 };
            let total = s.claims_total;
            s.push_ev(crate::engine::Ev { kind: 2, actor: caller as u8, target: accused as u8, n: revealed.len() as u8, total, flag, ..Default::default() });
        }
        s.push_ev(crate::engine::Ev { kind: 4, actor: p as u8, left: s.shots[p], flag: died as u8, ..Default::default() });
        let rev: Vec<&str> = revealed.iter().map(|&c| card_name(c)).collect();
        let cname = if caller >= 0 { s.name(caller as usize) } else { "?".into() };
        let aname = if accused >= 0 { s.name(accused as usize) } else { "?".into() };
        println!("  {YELLOW}{BOLD}LIAR!{RESET} {cname} called {aname}: {} [{}] -> {} pulls ({}/6){}",
            if devil { "the DEVIL, everyone else with cards shoots" } else if lie { "it was a LIE" } else { "it was TRUE" },
            rev.join(" "), s.name(p), s.shots[p],
            if died { " and DIES" } else { ", survives" });
    }

    /// The plugin repeats the table card on my turn and hand events. A round logged before the
    /// deal carries the old table card; if this one differs, it's the real one.
    fn check_table(&mut self, e: &Value) {
        let Some(t) = e["table"].as_i64() else { return };
        let s = &mut self.st;
        if t > 0 && t != s.table && s.in_round {
            println!("  {YELLOW}{BOLD}table card is {}{RESET}{YELLOW} (the round was logged before the deal and said {}){RESET}",
                card_name(t), card_name(s.table));
            s.table = t;
            s.start_match = s.start_hand.iter().filter(|&&c| c == t || c == JOKER).count() as u8;
            s.advised_at = None;
        }
    }

    fn on_turn(&mut self, p: usize, e: &Value) {
        if p >= self.st.n() {
            return;
        }
        self.scout(p);
        self.st.turn_since_pull = true;
        if let Some(prev) = self.st.prev_turn {
            if prev != p {
                self.st.vote_direction(prev, p);
                // Anyone between the last turn and this one who still holds cards was passed over.
                // Twice in a row means they're gone (left or AFK): take them out of the round.
                let order = self.st.turn_order();
                if let (Some(a), Some(b)) = (order.iter().position(|&q| q == prev), order.iter().position(|&q| q == p)) {
                    let len = order.len();
                    let mut k = (a + 1) % len;
                    while k != b {
                        let q = order[k];
                        let s = &mut self.st;
                        if q != s.me && s.alive[q] && s.hands[q] > 0 {
                            s.skipped[q] = s.skipped[q].saturating_add(1);
                            if s.skipped[q] == 2 {
                                println!("  {DIM}{} keeps getting skipped: probably left or AFK, treating them as out{RESET}", s.name(q));
                            }
                        }
                        k = (k + 1) % len;
                    }
                }
            }
        }
        self.st.skipped[p] = 0;
        self.st.prev_turn = Some(p);
        if p != self.st.me || !self.st.in_round {
            return;
        }
        // The game can flash my turn back on right after I play (e.g. when a departed player's
        // seat resets). I can never act on my own claim, so that is not a real turn.
        if self.st.last.is_some_and(|(q, _)| q == self.st.me) {
            return;
        }
        self.check_table(e);
        if e.get("hand").is_some() {
            let hand = ints(&e["hand"]);
            // Before my first play of the round, the turn's hand is my real deal: if it differs
            // from what the round logged (stale or still dealing), take it and advise again.
            if self.st.claims[self.st.me] == 0 && !hand.is_empty() && hand != self.st.start_hand {
                self.st.start_hand = hand.clone();
                self.st.hand = hand;
                self.st.start_match = self.st.my_match();
                self.st.start_devil = self.st.my_devil();
                self.st.advised_at = None;
            } else {
                self.st.hand = hand;
            }
        }
        self.advise_once();
    }

    /// Next player after `p` in turn order who is alive and still holds cards.
    fn next_holder(&self, p: usize) -> Option<usize> {
        let order = self.st.turn_order();
        let i = order.iter().position(|&q| q == p)?;
        (1..order.len()).map(|k| order[(i + k) % order.len()]).find(|&q| self.st.alive[q] && self.st.hands[q] > 0 && self.st.skipped[q] < 2)
    }

    fn advise_once(&mut self) {
        // Without my cards (still being dealt) nothing can be shown; don't mark the turn as
        // advised, so the turn signal that brings my hand still gets its advice.
        if self.st.advised_at == Some(self.st.claims_total) || self.st.hand.is_empty()
            || !self.st.alive.get(self.st.me).copied().unwrap_or(true) {
            return;
        }
        self.st.advised_at = Some(self.st.claims_total);
        self.advise();
    }

    /// The Devil policy once Devil mode is seen, otherwise the basic one.
    fn policy(&self) -> Arc<Policy> {
        match (&self.devil, self.st.devil_mode) {
            (Some(d), true) => d.clone(),
            _ => self.basic.clone(),
        }
    }

    fn advise(&mut self) {
        let policy = self.policy();
        let (g, me, order) = self.st.to_game();
        let s = &self.st;
        if s.hand.is_empty() {
            return;
        }
        let hand: Vec<String> = s.hand.iter().map(|&c| {
            let col = if c == DEVIL { MAGENTA } else if s.is_match(c) { GREEN } else { RED };
            format!("{col}{}{RESET}", card_name(c))
        }).collect();
        println!();
        println!("{}", rule("YOUR TURN"));
        println!("  table {BOLD}{}{RESET}   your hand  {}", card_name(s.table), hand.join(" "));
        // Devil's Deal comes first: the rest of the options share what's left.
        let mut deal_p = 0.0;
        if g.deal_legal() {
            let pa = deal_p_all_lies(&g, me);
            // Bots trained before Devil's Deal existed have no entry: show the card count only.
            let trained = match self.net_devil.as_ref() {
                Some(n) => Some(vec![0.0, crate::nn::Net::probs(&n.logits(&g, me, crate::nn::KIND_DEAL), 0, 2)[1]]),
                None => policy.call_probs(deal_key(&g, me)).map(|pr| pr.to_vec()),
            };
            deal_p = trained.as_ref().map_or(0.0, |pr| pr[1] as f64);
            // Guard: the trained Devil bot calls deals too often (won 8% of them against strong
            // opponents), so only pass its advice on when the card count gives a deal a real chance.
            if pa < 0.25 {
                deal_p = 0.0;
            }
            // Only worth showing when a deal has a real chance or the bot actually considers it.
            if pa >= 0.05 || deal_p >= 0.01 {
                let parts: Vec<String> = g.deal_window().map(|(pl, k)| format!("{} {k}", s.name(order[pl.player]))).collect();
                println!("  {DIM}Devil's Deal on the last 4 cards ({}): all lies ~{:.0}% by card count; win = everyone else with cards shoots, lose = you shoot{RESET}",
                    parts.join(" + "), pa * 100.0);
                if trained.is_some() {
                    println!();
                    println!("{}", bar_line(&format!("{MAGENTA}{BOLD}{:<26}{RESET}", "DEVIL'S DEAL"), deal_p, MAGENTA));
                } else {
                    println!("  {DIM}(this Devil bot predates Devil's Deal: no Deal advice yet){RESET}");
                }
            }
        }
        if g.is_forced(me) {
            println!();
            println!("  {YELLOW}{BOLD}You must call LIAR{RESET}  {DIM}(you are the only one with cards left){RESET}");
            if self.auto_play && !self.catching_up {
                let _ = std::fs::write(r"C:\liars\aq_action.txt", "call");
                println!("  {BOLD}[auto] call{RESET}");
            }
            println!("{}", rule(""));
            return;
        }
        let mut call_p = None;
        let generic = crate::policy::Feat { history: false, self_image: false, ..policy.feat() };
        let mut fb_note: Option<&str> = None;
        if let Some(lp) = g.last {
            let claimer = order[lp.player];
            let pt = p_claim_true(&g, me);
            // Unseen exact spot: ask the bot again treating opponents as unknown (well trained),
            // and only then the simple card-count rule.
            let net = if g.devil_mode { self.net_devil.clone() } else { self.net.clone() };
            let pc = match net.as_ref().map(|n| n.logits(&g, me, crate::nn::KIND_CALL)) {
                Some(out) => crate::nn::Net::probs(&out, 2, 2)[1] as f64,
                None => match policy.call_probs(call_key(&g, me, &policy.feat())) {
                Some(pr) => pr[1] as f64,
                None => match policy.call_probs(call_key(&g, me, &generic)) {
                    Some(pr) => { fb_note = Some("rare spot: advice ignores player history"); pr[1] as f64 }
                    None => {
                        fb_note = Some("never-seen spot: simple card-count rule, not the trained bot");
                        if self.fallback.decide_call(&g, me, &mut self.rng) { 1.0 } else { 0.0 }
                    }
                },
                },
            };
            // Ranked Devil lobbies bluff far less than the simulation the network trained in
            // (called plays were lies 27% on 2026-10-31 vs 47% simulated), so scale the call odds
            // by the prior-odds ratio until the retrained network replaces this.
            // LIARS_DEVIL_CALL_ODDS overrides it (1 = off).
            let pc = if g.devil_mode {
                let k: f64 = std::env::var("LIARS_DEVIL_CALL_ODDS").ok().and_then(|v| v.parse().ok()).unwrap_or(0.42);
                let o = pc / (1.0 - pc).max(1e-9) * k;
                o / (1.0 + o)
            } else { pc };
            self.eval_pend.set(Some((claimer, pc, pt, f64::NAN)));
            // Every spot where I face a claim (called or not): LIARS_SPOT_DUMP gets one line per
            // evaluation (consecutive repeats of a spot are dropped by the reader).
            if let Ok(path) = std::env::var("LIARS_SPOT_DUMP") {
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
                    use std::io::Write as _;
                    let _ = writeln!(f, "{},{},{},{},{},{:.5},{:.5},{},{}", s.name(claimer).replace(',', " "), lp.n, g.claims_total, g.hand(lp.player), g.n_alive(), pc, pt, g.hm[me], g.hand(me));
                }
            }
            // Only the last play is revealed. If their earlier claims were the bluffs, this play only
            // needs `lp.n` true cards in their hand; if those were honest, it needs all of them.
            let unseen = g.truths().saturating_sub(g.hm[me] + g.played_match[me]) as u64;
            let last_hi = hypergeom_at_least(lp.n as u64, (DECK_SIZE - HAND_SIZE) as u64, unseen, HAND_SIZE as u64);
            // Override: the networks learned that hand-emptying throws are nearly always true (the
            // simulated players rarely dump fakes) and pass even when the card count says the
            // throw can hardly be true. Real dumpers (Munk) fake it, so call those.
            let dump_override = g.hand(lp.player) == 0 && last_hi.max(pt) < 0.20;
            // A play the card count proves false is called (unless it could be the Devil card,
            // which punishes the caller).
            let certain = last_hi.max(pt) <= 1e-9 && (lp.n >= 2 || !devil_possible(&g, me));
            let pc = if dump_override || certain { 1.0 } else { pc };
            // Call rule v50 (2026-11-02): lie detector (logistic regression on 1053 labeled plays).
            // Replaces the card-count-only rule. On the training set: 76% precision at the same call
            // rate as the old rule (vs 35%), catches 129 lies the card count missed.
            // LIARS_CALL_RULE=0 restores the network's call odds. LIARS_LIE_DETECTOR=0 uses the
            // old card-count rule (last_hi.max(pt) < 0.20).
            let net_pc = pc;
            let call_rule = std::env::var("LIARS_CALL_RULE").unwrap_or_default();
            let rule_on = g.devil_mode && call_rule != "0" && call_rule != "data" && call_rule != "ab" && !call_rule.is_empty() || (g.devil_mode && std::env::var("LIARS_CALL_RULE").is_err());
            let data_on = g.devil_mode && self.data_call.is_some() && (call_rule == "data" || (call_rule == "ab" && self.ab_arm == Some(true)));
            let ld_on = std::env::var("LIARS_LIE_DETECTOR").map(|v| v != "0").unwrap_or(true);
            let last_think = s.round_thinks.last().map(|&(_, t)| t).unwrap_or(0.0);
            let ld_prob = if rule_on && ld_on && !(dump_override || certain) {
                lie_detector(&s, claimer, lp.n, last_think, pt, last_hi)
            } else { 0.0 };
            let ld_base: f64 = std::env::var("LIARS_LD_THRESH").ok().and_then(|v| v.parse().ok()).unwrap_or(0.65);
            // Per player: +10 points for every call of mine on them this game that showed real cards.
            // One step only (+10), and it clears the moment that player is caught lying: on 2026-11-03
            // three steps locked the detector out against a player bluffing triples every round.
            let vouched = s.vouched.get(claimer).copied().unwrap_or(0).min(1);
            let ld_threshold = (ld_base + 0.10 * vouched as f64).min(0.95);
            let cc_override_thresh: f64 = std::env::var("LIARS_CC_THRESH").ok().and_then(|v| v.parse().ok()).unwrap_or(0.15);
            let cc_override = last_hi.max(pt) <= cc_override_thresh && (lp.n >= 2 || !devil_possible(&g, me));
            let pc = if rule_on && !(dump_override || certain) {
                if cc_override {
                    1.0
                } else if ld_on {
                    if ld_prob >= ld_threshold { 1.0 } else { 0.0 }
                } else {
                    if last_hi.max(pt) < 0.20 { 1.0 } else { 0.0 }
                }
            } else { pc };
            // v61 (2026-11-06): data-fitted call model replaces the network call odds and every
            // card-count override except `certain`. Features as in research/call_scorer.py; the
            // claimer bluff prior comes from their labelled plays in earlier games.
            let mut data_note: Option<(f64, f64, f64)> = None;
            let pc = if data_on && !certain {
                let dc = self.data_call.as_ref().unwrap();
                let (prior, prior_n) = dc.prior_info(&s.name(claimer));
                let rc = s.game_called.get(claimer).copied().unwrap_or(0);
                let rl = s.game_lies.get(claimer).copied().unwrap_or(0);
                match crate::datacall::features(&g, me, last_think, prior, rc, rl) {
                    Some(f) => {
                        let p = dc.p_lie(&f);
                        data_note = Some((p, prior, prior_n));
                        if p >= dc.thresh { 1.0 } else { 0.0 }
                    }
                    None => pc,
                }
            } else { pc };
            // Endgame override (2026-11-01): only the claimer and I hold cards, they have ONE card left
            // after this claim, and I hold no real card and no Devil. Any bluff I make loses for sure
            // against a player who knows the rules: with a real card left they play it and I'm forced
            // to call it; with junk they must call me (playing it into my forced call is suicide), and
            // every call of my bluff lands. Calling now loses only if the card they just played is
            // real, so call whatever the count says. LIARS_LASTCARD_CALL=0 disables it.
            let last_card_spot = g.hand(lp.player) == 1 && g.hm[me] == 0 && g.hd[me] == 0 && g.hand(me) >= 1
                && !(0..g.n).any(|q| q != me && q != lp.player && g.active(q))
                && std::env::var("LIARS_LASTCARD_CALL").map(|v| v != "0").unwrap_or(true);
            let pc = if last_card_spot { 1.0 } else { pc };
            // 1v1 Devil holding no real cards (and no Devil card): the networks call at ~100% even
            // on claims the card count says are 99% true (learned in the simulator), but in the logs
            // calling there made me pull 77% of the time (36/47) while bluffing got caught 19%
            // (5/27). Cap the call at the card-count chance the play is a lie.
            // OFF by default (LIARS_ZERO1V1_CAP=1 turns it on): counted per round, calling first led to
            // my pull 75% (30/40) and bluffing first 69% (11/16), so the 77%-vs-19% above was an
            // artifact of only looking at the next play.
            let pc = if g.devil_mode && g.n_alive() == 2 && g.hm[me] == 0 && g.hd[me] == 0 && !(dump_override || certain)
                && std::env::var("LIARS_ZERO1V1_CAP").map(|v| v == "1").unwrap_or(false) {
                pc.min(1.0 - last_hi.max(pt))
            } else { pc };
            // v62 (2026-11-06): safe-hand guard. If my real cards (table card or joker) can carry me to the end of
            // the round, a call only adds risk: not calling is riskless. Round ends when the smallest hand at the
            // table empties, so "all my cards are real" or "real cards >= smallest other active hand" is safe.
            // Only a provably false claim (`certain`) is still called. LIARS_SAFE_HAND=0 disables.
            let min_other_hand = (0..g.n).filter(|&q| q != me && g.active(q)).map(|q| g.hand(q)).min().unwrap_or(0);
            // v62b: default is the narrow rule (all my cards real, or the claimer just emptied their hand while my
            // real cards cover the rest of the round: the round ends with no pull if I pass). LIARS_SAFE_HAND=wide
            // is the v62 rule (real cards >= smallest other hand), 0 is off. Logs 2026-11-06: the wide rule's extra
            // spots were 12 lies / 4 truths, the narrow spots 0 lies / 12 truths.
            let safe_mode = std::env::var("LIARS_SAFE_HAND").unwrap_or_else(|_| "1".to_string());
            let covers_round = g.hm[me] == g.hand(me) || g.hm[me] >= min_other_hand;
            // v62c: a hand-emptying throw is never called unless provably false (passing ends the round with no
            // pull; 1v1 dumps the bot called were true 92/92 times Nov 3-6, 3-4-player dumps 102/126).
            let dump_guard = g.devil_mode && safe_mode != "0" && g.hand(lp.player) == 0;
            let safe_hand = dump_guard || (g.devil_mode && g.hm[me] >= 1 && safe_mode != "0"
                && (g.hm[me] == g.hand(me)
                    || (safe_mode == "wide" && covers_round)));
            let safe_blocked = safe_hand && !certain && pc > 0.0;
            let pc = if safe_hand && !certain { 0.0 } else { pc };
            call_p = Some(pc * (1.0 - deal_p));
            let pace = match s.last_pace {
                Some("fast") => format!("  {GREEN}played FAST for them{RESET} {DIM}(weak tell: leans honest){RESET}"),
                Some("slow") => format!("  {YELLOW}played SLOW for them{RESET} {DIM}(weak tell: leans lie){RESET}"),
                Some("very slow") => format!("  {YELLOW}played VERY SLOW for them{RESET} {DIM}(weak tell: leans lie){RESET}"),
                Some(l) => format!("  {DIM}played {l} for them{RESET}"),
                None => String::new(),
            };
            println!("  {BOLD}{}{RESET} claimed {BOLD}{} {}{RESET}{pace}", s.name(claimer), lp.n, card_name(s.table));
            println!("  {DIM}card count: all their claims true {:.0}%  |  this play true: {:.0}-{:.0}% (depends which earlier claims were bluffs){RESET}",
                pt * 100.0, pt * 100.0, last_hi.max(pt) * 100.0);
            println!();
            println!("{}", bar_line(&format!("{YELLOW}{BOLD}{:<26}{RESET}", "CALL LIAR"), pc * (1.0 - deal_p), YELLOW));
            if let Some((p, prior, prior_n)) = data_note {
                let dc = self.data_call.as_ref().unwrap();
                println!("  {DIM}(data call model: {:.0}% lie, threshold {:.0}%; their bluff prior {:.0}% from {:.0} logged plays; network alone would say {:.0}%){RESET}",
                    p * 100.0, dc.thresh * 100.0, prior * 100.0, prior_n, net_pc * 100.0);
            }
            if rule_on && !(dump_override || certain) {
                if ld_on {
                    let vouch_note = if vouched > 0 { format!(" (+{} after {} real call{} on them)", vouched as u32 * 10, vouched, if vouched == 1 { "" } else { "s" }) } else { String::new() };
                    println!("  {DIM}(lie detector: {:.0}% lie probability, threshold {:.0}%{vouch_note}; network alone would say {:.0}%){RESET}", ld_prob * 100.0, ld_threshold * 100.0, net_pc * 100.0);
                } else {
                    println!("  {DIM}(card-count rule: calls when this play is under 20% true; the network alone would say {:.0}%){RESET}", net_pc * 100.0);
                }
            }
            let name = |k: usize| s.name(order[k]);
            for line in proof_lines(&g, me, lp.n, last_hi.max(pt), card_name(s.table), &name) {
                println!("  {line}");
            }
            if last_card_spot && !safe_blocked {
                println!("  {YELLOW}{BOLD}(endgame override: they have one card left and you hold no real card; any bluff is called or forced, so call now){RESET}");
            }
            if safe_blocked && dump_guard {
                println!("  {GREEN}{BOLD}(dump guard: they emptied their hand; passing ends the round with no pull, and called dumps were true 92/92 times 1v1 in the logs, so no call){RESET}");
            } else if safe_blocked {
                println!("  {GREEN}{BOLD}(safe-hand guard: you hold {} real card{} and can finish the round without lying; a call could only hurt, so no call){RESET}",
                    g.hm[me], if g.hm[me] == 1 { "" } else { "s" });
            }
            if certain {
                println!("  {YELLOW}{BOLD}(card-count override: this play can't be true){RESET}");
            } else if dump_override && !data_on {
                println!("  {YELLOW}{BOLD}(card-count override: they emptied their hand and the throw is under 20% likely true){RESET}");
            } else if cc_override && rule_on {
                println!("  {YELLOW}{BOLD}(card-count override: this play is under {:.0}% likely true){RESET}", cc_override_thresh * 100.0);
            }
        }
        let lg = &LEGAL[g.hm[me] as usize][g.hn[me] as usize];
        let na = g.n_actions(me);
        let act = |i: usize| if i < lg.n { lg.acts[i] } else { DEVIL_PLAY };
        let net = if g.devil_mode { self.net_devil.clone() } else { self.net.clone() };
        let probs: Vec<f32> = match net.as_ref().map(|n| n.logits(&g, me, crate::nn::KIND_PLAY)) {
            Some(out) => crate::nn::Net::probs(&out, 4, na),
            None => match policy.play_probs(play_key(&g, me, &policy.feat())) {
            Some(pr) => pr[..na].to_vec(),
            None => match policy.play_probs(play_key(&g, me, &generic)) {
                Some(pr) => { fb_note.get_or_insert("rare spot: advice ignores player history"); pr[..na].to_vec() }
                None => {
                    fb_note = Some("never-seen spot: simple card-count rule, not the trained bot");
                    // Real cards if I have them, else a plain one-card bluff: never mix the two.
                    let a = if g.hm[me] > 0 { (g.hm[me].min(3), 0) } else { (0, 1) };
                    (0..na).map(|i| if act(i) == a { 1.0 } else { 0.0 }).collect()
                }
            },
            },
        };
        if net.is_some() {
            println!("  {DIM}(neural network advice){RESET}");
        }
        // A mixed play (real + fake) is called a lie just like a pure bluff and spends a real card,
        // so when a pure bluff of the same size is possible, show its share on that bluff instead.
        let mut probs = probs;
        for i in 0..na {
            let a = act(i);
            if a != DEVIL_PLAY && a.0 > 0 && a.1 > 0 && g.hn[me] >= a.0 + a.1 {
                if let Some(j) = (0..na).find(|&j| act(j) == (0, a.0 + a.1)) {
                    probs[j] += probs[i];
                    probs[i] = 0.0;
                }
            }
        }
        // v63 candidate (LIARS_TRUTH_FIRST=1): a bluff of size k while holding >= k matching cards becomes the truthful
        // throw of size k (display and --auto action both follow the moved mass).
        if crate::agents::truth_first() {
            for i in 0..na {
                let a = act(i);
                if a != DEVIL_PLAY && a.1 > 0 && g.hm[me] >= a.0 + a.1 && probs[i] > 0.0 {
                    if let Some(j) = (0..na).find(|&j| act(j) == (a.0 + a.1, 0)) {
                        probs[j] += probs[i];
                        probs[i] = 0.0;
                    }
                }
            }
            println!("  {DIM}(truth-first: real cards instead of fakes when the hand allows){RESET}");
        }
        if let Some(n) = fb_note {
            println!("  {DIM}({n}){RESET}");
        }
        // Ordering rule v53 (2026-11-03 evening): junk until the junk is gone, then the real cards,
        // so the hand empties on a true claim. v52 defined "early" as the table's first two plays, so
        // at a 3-4 seat table the bot's second throw was already "late": it spent its real cards then
        // and was left holding junk for the last card, which got called about half the time (10 of 20
        // deaths between 13:40 and 18:00, 30 of 75 late bluffs called). v52 also never opened with a
        // true card in 60 games; one opening throw in four is now real when there are 2+ real cards.
        // LIARS_ORDER=0 disables it. LIARS_OPEN_TRUE sets the opening-truth share (default 0.25).
        if std::env::var("LIARS_ORDER").map(|v| v != "0").unwrap_or(true) {
            let real = g.hm[me];
            let junk = g.hn[me];
            let my_plays = s.round_plays.iter().filter(|&&(p, _, _, _)| p == s.me).count();
            // v57 (2026-11-04): the table learns the opener within a game. The bot's 2-junk opener was
            // called 13% in round 1 and 34% from round 4 on (humans stay near 10%), and 21%/32%/39%
            // after being caught 0/1/2+ times. So the real-opening share starts at LIARS_OPEN_TRUE
            // (0.25) and rises LIARS_OPEN_STEP (0.20) per exposure this game, capped at 0.65; and half
            // of the junk openers are a single card (bot 1-card openers: called 19% late in the game,
            // 2-card 30%, 3-card 46%).
            let open_base: f64 = std::env::var("LIARS_OPEN_TRUE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.25);
            let open_step: f64 = std::env::var("LIARS_OPEN_STEP").ok().and_then(|v| v.parse().ok()).unwrap_or(0.20);
            let open_true = (open_base + open_step * s.my_caught as f64).min(0.65);
            // LIARS_OPEN_ONE: share of junk openers that are a single card. v57 used 0.5: openers were
            // called less (16% late vs 36%) but the extra throws died later instead (21 later-throw
            // deaths in 65 games vs 13 in 62 under v54), so v58 sets it to 0 and keeps only the
            // adaptive real-opening share.
            let open_one: f64 = std::env::var("LIARS_OPEN_ONE").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
            let (roll, roll2) = {
                let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0) as u64;
                let mut x = ns ^ 0x9E37_79B9_7F4A_7C15;
                x ^= x >> 33; x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD); x ^= x >> 33;
                ((x % 10_000) as f64 / 10_000.0, ((x >> 20) % 10_000) as f64 / 10_000.0)
            };
            // v54 (2026-11-04): don't be the last one holding cards. Under v53 the bot was forced to call
            // the final claim 13 times in 25 games and it was true all 13 times: 13 free pulls. When some
            // opponent is down to 2 cards and I hold at least 2 more than them, dump 3 junk at once so the
            // real cards (which empty the hand on a true claim) come next turn, not two turns away.
            let my_cards = g.hand(me);
            let min_other = (0..g.n).filter(|&q| q != me && g.active(q)).map(|q| g.hand(q)).min();
            let threat = matches!(min_other, Some(m) if m <= 2 && my_cards >= m + 2);
            let want: Option<((u8, u8), String)> = if real > 0 && junk > 0 {
                if threat {
                    let f = junk.min(3).max(1);
                    Some(((0, f), format!("someone is down to {} card{}: dump {} junk now so the real cards empty the hand next turn", min_other.unwrap_or(0), if min_other == Some(1) { "" } else { "s" }, f)))
                } else if my_plays == 0 && real >= 2 && roll < open_true {
                    Some(((1, 0), format!("opening with a real card this time ({:.0}% of openings, caught {} time{} this game), junk next", open_true * 100.0, s.my_caught, if s.my_caught == 1 { "" } else { "s" })))
                } else if my_plays == 0 && junk >= 2 && roll2 < open_one {
                    Some(((0, 1), String::from("opening with one junk card this time (half of junk openers), real cards last")))
                } else {
                    // v54 default: two junk at a time, real cards last. (v55 tried three at once and
                    // was worse: 3-card openers called 59%; v56 shipped with that line by mistake.)
                    let f = junk.min(2).max(1);
                    Some(((0, f), String::from("junk first until it is gone, real cards last")))
                }
            } else if real > 0 && junk == 0 {
                Some(((real.min(3), 0), String::from("only real cards left: throw them all")))
            } else {
                None
            };
            if let Some((w, why)) = want {
                let mut pick = (0..na).find(|&j| act(j) == w);
                if pick.is_none() && w.1 > 2 { pick = (0..na).find(|&j| act(j) == (0, 2)); }
                if pick.is_none() && w.1 > 1 { pick = (0..na).find(|&j| act(j) == (0, 1)); }
                if pick.is_none() && w.0 > 1 { pick = (0..na).find(|&j| act(j) == (w.0 - 1, 0)); }
                if pick.is_none() && w.0 > 2 { pick = (0..na).find(|&j| act(j) == (1, 0)); }
                if let Some(j) = pick {
                    for p in probs.iter_mut() { *p = 0.0; }
                    probs[j] = 1.0;
                    println!("  {YELLOW}(ordering rule: {why}){RESET}");
                }
            }
        }
        // Hand-reader guard (2026-11-01): if the player who gets to call this play has been right on
        // 75%+ of 6+ calls, assume they see my cards. A bluff into them is a pull, so while I hold a
        // real card only real plays are offered. LIARS_READER_GUARD=0 disables it.
        if std::env::var("LIARS_READER_GUARD").map(|v| v != "0").unwrap_or(true) && g.hm[me] > 0 {
            if let Some(nx) = g.next_active(me) {
                let q = order[nx];
                if let Some((right, judged)) = s.sharp_caller(q) {
                    for i in 0..na {
                        let a = act(i);
                        if a != DEVIL_PLAY && a.1 > 0 { probs[i] = 0.0; }
                    }
                    let sum: f32 = probs.iter().sum();
                    if sum > 0.0 {
                        for p in probs.iter_mut() { *p /= sum; }
                    } else if let Some(j) = (0..na).find(|&j| act(j) == (1, 0)) {
                        probs[j] = 1.0;
                    }
                    println!("  {RED}{BOLD}⚠ {}'s calls have been right {right}/{judged}: treated as seeing your cards, no bluff while you hold a real card{RESET}", s.name(q));
                }
            }
        }
        if g.devil_active && !policy.devil {
            println!("  {DIM}(no Devil policy loaded: advice ignores the Devil){RESET}");
        }
        // Exact forced line: if all my cards are real and fit in one throw, and only one other
        // player would still hold cards, they are forced to call real cards and must pull.
        let my_cards = g.hm[me] + g.hn[me] + g.hd[me];
        let holders: Vec<usize> = (0..g.n).filter(|&q| q != me && g.alive[q] && g.hand(q) > 0).collect();
        if my_cards >= 1 && my_cards <= 3 && g.hn[me] == 0 && g.hd[me] == 0 && holders.len() == 1 {
            let victim = s.name(order[holders[0]]);
            let cards: Vec<&str> = s.hand.iter().map(|&c| card_name(c)).collect();
            println!("  {GREEN}{BOLD}★ PLAY [{}]{RESET}: empties your hand with real cards. {victim} is forced to call and must pull.",
                cards.join(" "));
        }
        // Play shares are shown as a share of all options, so they sum with CALL LIAR to 100%.
        let scale = 1.0 - deal_p - call_p.unwrap_or(0.0);
        let mut idx: Vec<usize> = (0..na).collect();
        idx.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        for &i in idx.iter().take(4) {
            let p = probs[i] as f64 * scale;
            if p < 0.005 {
                continue;
            }
            let a = act(i);
            let col = if a == DEVIL_PLAY { MAGENTA } else if a.1 == 0 { GREEN } else { RED };
            println!("{}", bar_line(&format!("{col}{:<26}{RESET}", self.short(a)), p, col));
        }
        // --auto: write the best action to a file for the mod's auto-play
        // Format: "play <real> <fake>" (matching and non-matching counts) or "call" or "deal"
        if self.auto_play && !self.catching_up {
            let best_play_i = probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i);
            let best_play_p = best_play_i.map_or(0.0, |i| probs[i] as f64 * scale);
            let best_call = call_p.unwrap_or(0.0);
            let mut action = if deal_p > best_call && deal_p > best_play_p {
                "deal".to_string()
            } else if best_call > best_play_p {
                "call".to_string()
            } else if let Some(i) = best_play_i {
                let a = act(i);
                if a == DEVIL_PLAY { "play devil".to_string() } else { format!("play {} {}", a.0, a.1) }
            } else {
                "play 0 1".to_string()
            };
            // Timing experiment (2026-11-01): real plays made slowly drew calls 4 of 4 times by hand
            // tonight, and across the logs plays off the table's tempo get believed while on-tempo
            // plays get called. True plays go slow most of the time; bluffs split between fast and
            // slow and avoid the middle; an occasional quick truth keeps "quick" from meaning bluff.
            // Measured on the bot's called-when-true (28%) and caught-bluff (25%) rates.
            // LIARS_TIMING=0 disables it (the plugin then uses its own 1.5-3.5 s).
            if std::env::var("LIARS_TIMING").map(|v| v != "0").unwrap_or(true) {
                let u = self.rng.f64();
                let r = self.rng.f64();
                // Capped 2026-11-03: across 5218 human-judged claims the call rate was 22% in every
                // think-time bucket, on the bot's claims too, so tempo buys nothing and the 16-24 s
                // bands only cost games per hour. Keep a little spread so plays are not metronomic.
                // LIARS_DELAY_MAX overrides the cap (seconds).
                let cap: f64 = std::env::var("LIARS_DELAY_MAX").ok().and_then(|v| v.parse().ok()).unwrap_or(5.0);
                let delay = if action.starts_with("play") {
                    if u < 0.6 { 1.5 + 1.5 * r } else { 3.0 + (cap - 3.0).max(0.0) * r }
                } else {
                    2.0 + (cap - 2.0).max(0.0) * r
                };
                // LIARS_DELAY_MIN (2026-11-07, user request): floor on every action's delay. The plugin fires ~1 s before
                // the advisor's delay, so 4.5 here means no visible action under about 3.5 s.
                let min_d: f64 = std::env::var("LIARS_DELAY_MIN").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
                let delay = delay.max(min_d);
                action = format!("{action} delay={delay:.1}");
            }
            let _ = std::fs::write(r"C:\liars\aq_action.txt", &action);
            println!("  {BOLD}[auto] {action}{RESET}");
        }

        println!("{}", rule(""));
    }

    /// Reads the other hands from this round's claims and plays every option out (see reason.rs).
    fn why(&self, g: &Game, me: usize, order: &[usize]) {
        if self.catching_up || std::env::var("LIARS_NO_WHY").is_ok() {
            return;
        }
        let s = &self.st;
        let model = if g.devil_mode { self.read_net_devil.clone() } else { self.read_net.clone() };
        let Some(model) = model else { return };
        let pos = |q: usize| order.iter().position(|&x| x == q);
        let plays: Vec<crate::reason::RoundPlay> = s.round_plays.iter().filter_map(|&(p, n, t, d)| pos(p).map(|k| (k, n, t, d))).collect();
        let sd = s.start_devil as u8;
        let size = (s.start_hand.len() as u8).max(s.start_match + sd);
        let my_start = (s.start_match, size - s.start_match - sd, sd);
        if std::env::var("LIARS_WHY_DEBUG").is_ok() {
            println!("  [why debug] plays {:?} my_start {:?} me {} hm {} hn {} hd {}", plays, my_start, me, g.hm[me], g.hn[me], g.hd[me]);
        }
        let t0 = std::time::Instant::now();
        let rollouts = std::env::var("LIARS_WHY_ROLLOUTS").ok().and_then(|v| v.parse().ok()).unwrap_or(200usize);
        let an = crate::reason::analyse(g, me, &plays, my_start, &model, crate::engine::StarterRule::Loser, 1500, rollouts);
        if an.options.is_empty() {
            return;
        }
        println!("  {BOLD}WHY{RESET} {DIM}(each option played to the end {}x from likely hands, {:.1}s){RESET}", rollouts, t0.elapsed().as_secs_f64());
        // The lie read itself was coin-flip quality on real players, so it is only logged for
        // testing (LIARS_EVAL_DUMP), not shown.
        if let Some((c, pc, pt, _)) = self.eval_pend.get() {
            self.eval_pend.set(Some((c, pc, pt, an.lie)));
        }
        let mut opts = an.options.clone();
        opts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let best = opts[0].1;
        let txt: Vec<String> = opts.iter().take(4).map(|(o, w)| {
            let name = match o { None => "CALL".to_string(), Some(a) => self.short(*a) };
            let t = format!("{name} {:.0}%", w * 100.0);
            if *w == best { format!("{BOLD}{t}{RESET}") } else { t }
        }).collect();
        println!("  win game if: {}", txt.join("  |  "));
        if opts.len() > 1 && best - opts[1].1 < 0.03 {
            println!("  {DIM}(top options within 3 points: a close call either way){RESET}");
        }
    }

    /// Compact label for a bar: "PLAY [Q Q]  truth".
    fn short(&self, a: (u8, u8)) -> String {
        if a == DEVIL_PLAY {
            return "PLAY [Devil] bait".to_string();
        }
        let d = self.describe(a);
        let cards = d.split('[').nth(1).and_then(|x| x.split(']').next()).unwrap_or("");
        let kind = match a {
            (_, 0) => "truth",
            (0, _) => "bluff",
            _ => "mixed",
        };
        format!("PLAY [{cards}] {kind}")
    }

    fn describe(&self, (t, b): (u8, u8)) -> String {
        let s = &self.st;
        if (t, b) == DEVIL_PLAY {
            return "the DEVIL alone (if called, everyone else with cards shoots)".to_string();
        }
        let mut real: Vec<i64> = s.hand.iter().copied().filter(|&c| c == s.table).collect();
        real.extend(s.hand.iter().copied().filter(|&c| c == JOKER)); // spend table cards before Jokers
        let fake: Vec<i64> = s.hand.iter().copied().filter(|&c| !s.is_match(c) && c != DEVIL).collect();
        let mut cards: Vec<&str> = real.iter().take(t as usize).map(|&c| card_name(c)).collect();
        cards.extend(fake.iter().take(b as usize).map(|&c| card_name(c)));
        let kind = match (t, b) {
            (_, 0) => "truth",
            (0, _) => "bluff",
            _ => "mixed",
        };
        format!("{} card{} [{}] ({kind})", t + b, if t + b > 1 { "s" } else { "" }, cards.join(" "))
    }
}

fn ints_f(v: &Value) -> Vec<f64> {
    v.as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).collect()).unwrap_or_default()
}

fn newest_log(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("game_") && n.ends_with(".jsonl")))
        .max_by_key(|p| p.metadata().and_then(|m| m.modified()).ok())
}

/// Follows the newest log in `dir`, switching when a newer one appears.
/// With `once`, reads what is there and exits (for checking a finished game).
#[cfg(windows)]
fn disable_quick_edit() {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetStdHandle(n: u32) -> isize;
        fn GetConsoleMode(h: isize, mode: *mut u32) -> i32;
        fn SetConsoleMode(h: isize, mode: u32) -> i32;
    }
    const STD_INPUT_HANDLE: u32 = 0xFFFF_FFF6; // (DWORD)-10
    const ENABLE_QUICK_EDIT_MODE: u32 = 0x0040;
    const ENABLE_EXTENDED_FLAGS: u32 = 0x0080;
    unsafe {
        let h = GetStdHandle(STD_INPUT_HANDLE);
        let mut mode = 0u32;
        if h != 0 && h != -1 && GetConsoleMode(h, &mut mode) != 0 {
            SetConsoleMode(h, (mode | ENABLE_EXTENDED_FLAGS) & !ENABLE_QUICK_EDIT_MODE);
        }
    }
}
#[cfg(not(windows))]
fn disable_quick_edit() {}

pub fn run(policy: Arc<Policy>, devil: Option<Arc<Policy>>, chaos: Option<Arc<ChaosPolicy>>, dir: &str, once: bool, auto: bool) {
    // 2026-11-05: a text selection in the console window (QuickEdit) blocks every println!, and the
    // advisor only writes aq_action.txt after printing its advice, so one click-drag in the window
    // silently froze the auto-play for a whole game (the mod fell back to playing one matching card).
    // Turn QuickEdit off for this console. Mark mode (Ctrl+M) still allows deliberate copying.
    disable_quick_edit();
    let dir = Path::new(dir);
    let mut w = Watcher::new(policy, devil, chaos);
    w.auto_play = auto;
    if auto { println!("auto-play: writing actions to C:\\liars\\aq_action.txt"); }
    // The Basic neural network, if one has been trained and passed its tests.
    let net_path = std::env::var("LIARS_NET").unwrap_or_else(|_| "C:/liars/nn/basic_ppo2".to_string());
    match crate::nn::Model::load(&net_path) {
        Ok(n) => {
            println!("Basic: {} {net_path}", if n.is_transformer() { "transformer network" } else { "neural network" });
            w.net = Some(Arc::new(n));
        }
        Err(_) => println!("Basic: table bot (no neural network at {net_path})"),
    }
    let devil_path = std::env::var("LIARS_DEVIL_NET").unwrap_or_else(|_| "C:/liars/nn/devil_live".to_string());
    match crate::nn::Model::load(&devil_path) {
        Ok(n) => {
            println!("Devil: {} {devil_path}", if n.is_transformer() { "transformer network" } else { "neural network" });
            w.net_devil = Some(Arc::new(n));
        }
        Err(_) => println!("Devil: table bot (no neural network at {devil_path})"),
    }
    for (var, def, slot) in [("LIARS_READ_NET", "C:/liars/nn/basic_live", 0), ("LIARS_READ_DEVIL_NET", "C:/liars/nn/devil_live", 1)] {
        let path = std::env::var(var).unwrap_or_else(|_| def.to_string());
        if let Ok(m) = crate::nn::Model::load(&path) {
            if !m.is_transformer() {
                if slot == 0 { w.read_net = Some(Arc::new(m)); } else { w.read_net_devil = Some(Arc::new(m)); }
            }
        }
    }
    if std::env::var("LIARS_CALL_RULE").map(|v| v == "data" || v == "ab").unwrap_or(false) {
        let path = std::env::var("LIARS_CALL_DATA").unwrap_or_else(|_| "C:/liars/research/call_model_v61.json".to_string());
        match crate::datacall::DataCall::load(&path) {
            Ok(dc) => {
                println!("Call rule: data model {path} (threshold {:.0}%, {} player priors)", dc.thresh * 100.0, dc.priors.len());
                w.data_call = Some(Arc::new(dc));
            }
            Err(e) => println!("Call rule: data model requested but not loaded ({e}); using the network call odds"),
        }
    }
    let mut current: Option<PathBuf> = None;
    let mut reader: Option<BufReader<File>> = None;
    let mut buf = String::new();
    let mut catch_up_to = 0u64;
    println!("watching {} (Ctrl+C to stop)", dir.display());
    loop {
        if let Some(newest) = newest_log(dir) {
            if current.as_ref() != Some(&newest) {
                // Read from the top so the rebuilt state covers the whole game.
                let f = File::open(&newest).expect("open log");
                println!("following {}", newest.display());
                let (net, net_devil) = (w.net.clone(), w.net_devil.clone());
                let (rn, rnd) = (w.read_net.clone(), w.read_net_devil.clone());
                let ap = w.auto_play;
                let dc = w.data_call.clone();
                w = Watcher::new(w.basic.clone(), w.devil.clone(), w.chaos_policy.clone());
                w.auto_play = ap;
                w.data_call = dc;
                // A/B: pick this game's call arm from the log file name (reproducible), record it.
                if std::env::var("LIARS_CALL_RULE").map(|v| v == "ab").unwrap_or(false) && w.data_call.is_some() {
                    let stem = newest.file_stem().and_then(|x| x.to_str()).unwrap_or("").to_string();
                    let mut h: u64 = 0xcbf29ce484222325;
                    for b in stem.bytes() { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); }
                    let data_arm = (h >> 7) & 1 == 1;
                    w.ab_arm = Some(data_arm);
                    println!("A/B arm for this game: {}", if data_arm { "DATA call model" } else { "NETWORK call odds" });
                    if !once {
                        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("C:/liars/research/ab_arms.jsonl") {
                            use std::io::Write as _;
                            let _ = writeln!(f, "{{\"game\":\"{}\",\"arm\":\"{}\"}}", stem.trim_start_matches("game_"), if data_arm { "data" } else { "net" });
                        }
                    }
                }
                w.net = net;
                w.net_devil = net_devil;
                w.read_net = rn;
                w.read_net_devil = rnd;
                w.profiles = build_profiles(dir, Some(newest.as_path()));
                w.track.out = w.profiles.iter().map(|(n, p)| (n.clone(), p.hab)).collect();
                catch_up_to = if once { 0 } else { f.metadata().map(|m| m.len()).unwrap_or(0) };
                w.catching_up = catch_up_to > 0;
                reader = Some(BufReader::new(f));
                current = Some(newest);
            }
        }
        let mut got = false;
        if let Some(r) = reader.as_mut() {
            loop {
                buf.clear();
                match r.read_line(&mut buf) {
                    Ok(0) => break,
                    Ok(n) if buf.ends_with('\n') => {
                        got = true;
                        // Turns already in the log when it was opened are replayed silently.
                        catch_up_to = catch_up_to.saturating_sub(n as u64);
                        w.catching_up = catch_up_to > 0;
                        if let Ok(v) = serde_json::from_str::<Value>(buf.trim()) {
                            if v["e"] != "pose" && v["e"] != "anchor" {
                                w.handle(&v);
                            }
                        }
                    }
                    Ok(n) => {
                        // Partial line: rewind and wait for the rest.
                        let back = n as i64;
                        r.seek_relative(-back).ok();
                        break;
                    }
                    Err(_) => break,
                }
            }
        }
        if once && !got {
            return;
        }
        if !got {
            std::thread::sleep(Duration::from_millis(100));
        }
    }
}


/// The Devil card could still be played by someone else this round (it isn't mine and hasn't
/// shown up).
fn devil_possible(g: &Game, me: usize) -> bool {
    g.devil_active && g.hd[me] == 0 && !g.played_devil.iter().any(|&d| d)
}

/// What my hand proves about the last play, the only one that can be called (by the next
/// player). On my turn: whether it can't be true. On someone else's: what they would need to hold
/// to know it's a lie once I tell them my count, and how likely they hold that.
fn proof_lines(g: &Game, me: usize, n: u8, this_hi: f64, card: &str, name: &dyn Fn(usize) -> String) -> Vec<String> {
    let mut out = Vec::new();
    let Some(lp) = g.last else { return out };
    let mine = g.hm[me] + g.played_match[me];
    let avail = g.truths().saturating_sub(mine);
    let devil = if n == 1 && devil_possible(g, me) { " (unless it's the Devil card)" } else { "" };
    if this_hi <= 1e-9 {
        out.push(format!("{YELLOW}{BOLD}100% LIE{RESET}{YELLOW}: only {avail} real {card} can be outside your hand{devil}{RESET}"));
        return out;
    }
    let Some(next) = g.next_active(lp.player) else { return out };
    if next == me || mine == 0 {
        return out;
    }
    // The play needs n real cards among the `avail` outside my hand; if the next player holds k
    // of those themselves, it's impossible once k > avail - n.
    let k = avail.saturating_sub(n) + 1;
    let their_hand = g.hand(next);
    if k > their_hand {
        return out;
    }
    let pop = (DECK_SIZE - HAND_SIZE) as u64;
    let chance = hypergeom_at_least(k as u64, pop, avail as u64, their_hand as u64);
    // Long shots only clutter the screen.
    if chance < 0.10 {
        return out;
    }
    // Said as a threshold for them, not my count (they'd have to work it back from the deck).
    let hold = if k == 1 { format!("even one {card}") } else { format!("{k} or more {card}s") };
    out.push(format!("{YELLOW}{BOLD}say{RESET}{YELLOW}: \"{}, if you have {hold}, call {}, it's a lie\"{devil}  {DIM}(chance they do: {:.0}%){RESET}",
        name(next), name(lp.player), chance * 100.0));
    out
}

/// One claim, for the habit counts.
#[derive(Clone, Copy)]
struct TClaim {
    p: i64,
    n: usize,
    opening: bool,
    empty: bool,
    duel: bool,
}

/// Counts the habits in engine::Prof from raw log events, by player name, the same way the
/// engine counts them in training (past logs and the live game alike). A play is held until the
/// next event, since a call logs the caller's hand being cleared as a play (see build_profiles).
#[derive(Default)]
pub struct ProfTrack {
    pub out: HashMap<String, crate::engine::Prof>,
    names: HashMap<i64, String>,
    alive_n: usize,
    seq: u32,
    rd: HashMap<i64, (u8, u8)>,
    last: Option<TClaim>,
    staged: Option<TClaim>,
    table: i64,
    called_this_round: bool,
    chaos: bool,
}

impl ProfTrack {
    fn prof(&mut self, raw: i64) -> Option<&mut crate::engine::Prof> {
        let n = self.names.get(&raw)?.clone();
        Some(self.out.entry(n).or_default())
    }

    fn commit(&mut self) {
        let Some(c) = self.staged.take() else { return };
        if self.last.is_some_and(|l| l.opening) {
            if let Some(pr) = self.prof(c.p) {
                pr.open_opps += 1;
            }
        }
        let (k, first) = self.rd.get(&c.p).copied().unwrap_or((0, 0));
        let dump = k == 1 && c.empty && first as usize + c.n == HAND_SIZE as usize;
        if let Some(pr) = self.prof(c.p) {
            pr.claims += 1;
            pr.claims3 += (c.n == 3) as u16;
            pr.dumps += dump as u16;
        }
        self.rd.insert(c.p, (k + 1, if k == 0 { c.n as u8 } else { first }));
        self.seq += 1;
        self.last = Some(c);
    }

    pub fn feed(&mut self, e: &Value) {
        let kind = e["e"].as_str().unwrap_or("");
        if kind == "start" {
            self.chaos = e["mode"].as_str() == Some("chaos");
        }
        if self.chaos {
            return;
        }
        let raw = e["p"].as_i64().unwrap_or(-1);
        if matches!(kind, "turn" | "play" | "round" | "seat" | "dead") {
            self.commit();
        }
        match kind {
            "seat" => {
                self.names.insert(raw, e["name"].as_str().unwrap_or("?").to_string());
            }
            "round" => {
                self.staged = None;
                self.last = None;
                self.seq = 0;
                self.rd.clear();
                self.called_this_round = false;
                self.table = e["table"].as_i64().unwrap_or(0);
                let alive = ints(&e["alive"]);
                self.alive_n = alive.iter().filter(|&&a| a != 0).count();
                for (q, &a) in alive.iter().enumerate() {
                    if a != 0 {
                        if let Some(pr) = self.prof(q as i64) {
                            pr.rounds += 1;
                        }
                    }
                }
            }
            "play" => {
                let n = e["n"].as_u64().unwrap_or(1) as usize;
                let left = e["left"].as_i64().unwrap_or(5);
                self.staged = Some(TClaim { p: raw, n, opening: self.seq == 0, empty: left == 0, duel: self.alive_n == 2 });
            }
            "pull" if !self.called_this_round => {
                self.called_this_round = true;
                let revealed = ints(&e["revealed"]);
                let lie = revealed.iter().any(|&c| c != self.table && c != JOKER);
                let mut caller = e["caller"].as_i64().unwrap_or(-1);
                // The hand-clear correction of build_profiles: the newest "play" may be the
                // caller's hand being cleared rather than a claim.
                let mut accused = self.staged.or(self.last);
                if let (Some(lp), Some(pp)) = (self.staged, self.last) {
                    let mismatch = revealed.len() != lp.n && revealed.len() == pp.n;
                    let puller = revealed.len() == pp.n && ((!lie && raw == lp.p) || (lie && raw == pp.p));
                    if lp.p != pp.p && (mismatch || puller) {
                        caller = lp.p;
                        accused = Some(pp);
                        self.staged = None;
                    }
                }
                self.commit();
                let Some(a) = accused else { return };
                if a.opening {
                    if let Some(pr) = self.prof(caller) {
                        pr.open_opps += 1;
                        pr.open_calls += 1;
                    }
                }
                if revealed.is_empty() || revealed.contains(&DEVIL) {
                    return;
                }
                if let Some(pr) = self.prof(a.p) {
                    if a.empty {
                        pr.empty_called += 1;
                        pr.empty_lies += lie as u16;
                    }
                    if a.duel {
                        pr.duel_called += 1;
                        pr.duel_lies += lie as u16;
                    }
                }
            }
            _ => {}
        }
    }
}
