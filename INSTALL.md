# Running the bot

Windows only. Two programs cooperate: a MelonLoader mod inside the game, and the advisor, a console program next to it.
They talk through a folder, `C:\liars`:

| file | written by | read by | what it is |
|---|---|---|---|
| `C:\liars\logs\game_<time>.jsonl` | mod | advisor | one event log per game: seats, claims, think times, calls, pulls, your own hand |
| `C:\liars\aq_action.txt` | advisor | mod | the next action: `play <real> <fake> delay=<s>`, `call`, `deal`, `play devil` |
| `C:\liars\aq_heartbeat.txt` | mod | watchdog | current auto-queue state, once a second |
| `C:\liars\aq_autostart.txt` | you | mod | if present, the mod starts queueing by itself 10 s after the game loads |
| `C:\liars\aq_ranked.txt` | you | mod | if present, queue ranked matchmaking instead of the public lobby list |
| `C:\liars\aq_restart.txt` | mod | watchdog | the mod asks for a game restart |

## 1. Install MelonLoader

Download the MelonLoader installer from its GitHub releases, point it at `Liar's Bar.exe` in your Steam folder, and
install the latest version. Start the game once and quit; this generates the `MelonLoader` and `Mods` folders.

## 2. Install the mod

Copy `LiarsTells.dll` into `<game folder>\Mods\`. Create `C:\liars\logs`. If you want the bot to queue on its own,
create empty files `C:\liars\aq_autostart.txt` and, for ranked, `C:\liars\aq_ranked.txt`. Without them the mod only
logs games and waits for the advisor's actions when you are seated at a table.

## 3. Run the advisor

Unzip the release anywhere, keep `liars.exe`, `weights\` and `run_advisor.bat` together, and double-click
`run_advisor.bat`. It tails `C:\liars\logs`, prints what it sees and what it would do, and writes the action file.
Variants are environment variables set in the bat; the defaults are the configuration that played ranked.

## 4. Start the game

Start Liar's Bar through Steam. With the autostart flag the mod clicks through to matchmaking by itself; otherwise sit at
a table and the mod will act on the advisor's output when it is your turn.

## Optional: the watchdog

`watchdog.py` (Python 3) restarts the game when the heartbeat goes stale or the mod asks, and relaunches the advisor from
`C:\liars\run_current_advisor.bat` when it is not running. Copy `run_advisor.bat` to that path to use it.

## Building from source

- Advisor and simulator: `cd rust && cargo build --release` (Rust stable). `target/release/liars.exe`.
- Mod: `cd mod && dotnet build -c Release` (.NET 6 SDK). The project resolves the game's assemblies from the default
  Steam path; set `GamePath` in the project file if yours differs. Output in `mod/bin/Release/net6.0/LiarsTells.dll`.
- Trainers: build `pyliars/` with `maturin develop --release`, then the scripts in `nn/`.

## Watching without acting

Run `liars.exe watch C:\liars\logs` without `--auto`. The advisor then only prints its reading of each spot and its
advice, and the mod never receives an action file, so you play yourself with the analysis alongside.
