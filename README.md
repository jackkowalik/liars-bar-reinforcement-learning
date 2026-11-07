# Liar's Bar ranked bot

A bot that plays Liar's Deck (Devil mode) in the ranked queue of Liar's Bar using only what a human player at the table can
see. It went from a 0 rating to the leaderboard in about 24 hours (~3000 rating points). 

## Components

- **Game engine and simulator** (`rust/src/engine.rs`): Liar's Deck with Devil mode, 6-chamber revolver, forced calls,
  Devil's Deal, 2 to 4 seats. Millions of games per hour on a desktop.
- **Policies** (`rust/src/nn.rs`, `nn/`): an MLP trained with PPO in self-play against a league of scripted opponents, and a
  transformer that reads the full round history with seat identities, distilled from the MLP and then trained with PPO.
  Both export to a flat weights file the Rust code evaluates with no dependencies.
- **Game-side mod** (`tells/`, MelonLoader, C#): writes a public event log for every game (seats, tiers, claims, think
  times, calls, pulls, our own hand) and executes the advisor's chosen action through the game's own input methods.
- **Advisor** (`rust/src/watch.rs`): tails the log, rebuilds the game state, runs the policy, prints its reasoning and
  writes the action for the mod. Every policy switch is one environment variable.
- **Data pipeline** (`nn/reveals.py`, `data/`): labels every opponent throw as true or bluff after the session, giving
  ground truth for all opponent plays (about 10,000 throws and 4,400 called claims). Labels are used offline only, never
  by the live advisor.
- **Fitted opponent population** (`research/fit_population.py`, `agents::FitHuman`): bluff rate, call rate, throw sizes
  and dump habits fitted to the labelled logs, with per-game variation.
- **Harness** (`liars fiteval`): any policy variant against the fitted population, thousands of games in seconds, same
  seed for A and B. Every change goes through it before going live.
- **Live measurement** (`research/scoreboard.py`): win rate from the game's own rating counters, by table size and
  opponent tier, with each variant's games kept separate and pre-registered sample sizes.

## Method

1. Train a policy in self-play in the simulator.
2. Play it live, log everything, label opponent throws from the logs.
3. Fit the opponent population to the labels and put it in the simulator.
4. Evaluate every candidate change against that population with a fixed seed before it touches the game.
5. Run the survivors live, one variable at a time, with a pre-registered sample size, a stop rule and a scoreboard that
   keeps each variant's games separate.

## Results

Live win rate from the game's rating counters: 47-50% over 130 ranked games at 2, 3 and 4-seat tables. Random play at the
observed table mix is 31%. Against the fitted population in the harness the same policy wins 48% (4 seats), 51% (3) and
67% (1v1).

Where the bot's deaths come from, 240 deaths from the logs:

| cause | deaths |
|---|---|
| bluff thrown while holding a matching card, called (avoidable in theory) | 74 |
| partial bluff (real and fake together), called | 60 |
| voluntary wrong call | 44 |
| forced call on a true hand-emptying throw (1v1) | 31 |
| bluff with no matching card, called | 20 |

Other measurements from the logs:

- Humans call about 22% of claims regardless of the claimer's think time, and call the bot's truthful throws (37%) more
  often than its bluffs (24%).
- Opponent tier does not change how often the bot's bluffs are called.
- Matchmaking is not rating-based: the opponent tier mix is flat, so the climb is linear in games
  played at a fixed payout! Huge
- The bot's own calls are right 37% of the time: opening-claim calls 62%, calls on the third to fifth claim of a round
  17-28%, forced calls 0%.

## Related work

- Stanford CS224R project, *Bluff and Learn*: CFR and NFSP agents on Liar's Bar; tabular CFR works in simplified versions,
  NFSP with dense per-round rewards generalises better at full size.
- *Outbidding and Outbluffing Elite Humans* (arXiv 2511.03724, 2025): Solly, a self-play R-NaD agent for reduced-format
  Liar's Poker at elite human level. It bluffs far more than humans do, consistent with heavy bluffing being correct when
  it is unexploitable, and with our finding that the problem is not the bluff rate but who the bluff is aimed at.
- *Social Behaviour and Strategic Adaptation of LLMs in Multiplayer Sequential Games* (NeurIPS 2025): a prompt-based
  Liar's Bar variant used to study LLM agents.
- Neller and Lanctot, *An Introduction to Counterfactual Regret Minimization*; Neller and Hnath on Dudo with imperfect
  recall abstraction: the classic treatment of bluffing games and why exact CFR does not scale to this one.

## Scope and conduct

The bot acts on public information only: claims, counts, timings and outcomes any player sees, plus its own hand. The
game client receives every player's hand from the server; that leak was used to label logged data after the fact and was
deliberately kept out of the live advisor; the labelled hands are released in `data/`, the code that read them is not. The bot played ranked
games against people who did not know they were playing against a machine, which the game's rules almost certainly prohibit.

## Dataset and weights

`data/plays_public.jsonl` has 15,314 claims from 522 games, each labelled true or bluff with every seat's hidden hand
at the moment of the claim, the claimer's hand and the cards actually thrown (5,616 unlabelled claims from 194 earlier games are in
`data/plays_unlabelled.jsonl`). `data/calls_public.jsonl`
has the 4,439 claims that were called, with the revealed cards and the same hidden hands. Player names are pseudonymised
consistently across both files; see `data/SCHEMA.md`. `weights/` holds the networks the Rust code runs: `devil_mlp` (the one that played ranked),
`devil_tnet_r1v1` (the transformer) and `basic_ppo2` (non-Devil tables).

```
cd rust && cargo build --release
target/release/liars fiteval --net ../weights/devil_mlp --opp rankedfit --players 4 --games 3000
```

## Layout

```
rust/        engine, simulator, policies, advisor, harness
pyliars/     Python binding to the engine, used by the trainers
nn/          training (PPO, distillation), population fitting
weights/     exported networks
mod/         MelonLoader mod: event log, auto-queue state machine, action execution
tools/       watchdog, advisor launcher, idle-time restart
research/    population fit, scoreboard, call scorer, bluff pricing, dataset export
data/        the released dataset and its schema
```

## License

MIT. The dataset in `data/` is released under the same license.
