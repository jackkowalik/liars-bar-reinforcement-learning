# Dataset

Two JSONL files exported by `research/export_dataset.py` from the bot's game logs (ranked Devil-mode tables, 2 to 4
seats). Player names are replaced by stable pseudonyms (salted SHA-1 prefix, `p…`), game ids likewise (`g…`); the same
pseudonym refers to the same player across both files and across games. Every seat's hidden hand at the time of each
labelled claim is included (`hands`), along with the claimer's hand and the cards actually thrown: this is the ground truth
behind `was_bluff` and lets you model what players do given what they hold. The bot's own seat name varied between
sessions, so identify the bot by the `*_is_bot` flags, not by pseudonym.

Cards: `K`, `Q`, `A`, `Joker` (matches any table card), `Devil` (Devil-mode card). `table` is the round's table card.

## plays_public.jsonl (20,930 rows; 15,314 with a label)

One row per claim (throw) by any player.

| field | meaning |
|---|---|
| game, t | game pseudonym, seconds since the client started |
| mode, ranked, seats | "Devil"; ranked flag when known; players at the table |
| round, table, n_alive | round number, table card, players alive at the claim |
| players | one entry per seat: `seat`, `player` pseudonym, `alive`, `cards` in hand, `pulls` so far, `is_bot`, `tier`, `level` |
| claims_before | earlier claims this round: seat `p`, cards `n`, cards `left` after, `think` seconds, `index` in round |
| play | this claim, same fields plus `player` and `earlier` (that player's earlier claims this round) |
| play_is_bot | true when the bot made the claim |
| bot_hand, bot_matching | the bot's hand at the time and how many of its cards matched the table card |
| claimer_hand | the claimer's full hand before the throw (labelled rows) |
| thrown | the cards actually thrown (13,273 rows; null where only the count could be inferred) |
| hands | every seat's hand at the time, keyed by player pseudonym (labelled rows) |
| was_bluff | true if any thrown card did not match the table card (Joker counts as matching); null when unlabelled |
| label_source | "diff" (hand change observed) or "count" (inferred from counts) |

## calls_public.jsonl (4,439 rows)

One row per claim that was called.

| field | meaning |
|---|---|
| game, t, mode, ranked, seats, round, table, n_alive, players, hands | as above |
| claims | all claims of the round up to and including the called one |
| claim | the called claim |
| caller, caller_is_bot, claimer_is_bot | who called; bot flags |
| lie | whether the called claim was a lie |
| revealed | the cards revealed by the call (public) |
| pulled | who pulled the trigger and the outcome, as logged |
| devil | Devil's Deal information when the call was one |
| bot_hand, bot_matching | the bot's hand at the time |
