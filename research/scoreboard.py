"""v60 scoreboard: win rate from the game's own rating counters (consecutive game-start mmr lines),
restricted to games started after the v60 swap. Usage: python scoreboard.py [since YYYYMMDD_HHMMSS]
The pre-registered bar is a win rate > 0.228 over >= 200 games (RESEARCH.md, 2026-11-06 04:10)."""
import json, glob, os, sys, math
BAR = 0.228
# games lost to our own tooling, not to play (excluded from the tally): 14:08:34 the watchdog killed the game at the
# start of round 1 because a DLL-swap waiter treated WaitingForGame as idle.
EXCLUDE = {"20261106_140834"}
V62_SINCE = "20261106_215031"  # v62 safe-hand guard live (research/swap_v62.log); A/B paused
V60_RESUME = "20261106_221522"
# v63 truth-first trial: starts at the time swap_v63.log reports the new advisor running (None until the swap fires)
V63_SINCE = None
try:
    for l in open("C:/liars/research/swap_v63.log", encoding="utf8"):
        if "advisor running pid" in l: V63_SINCE = "20261107_" + l[:8].replace(":", "")
except FileNotFoundError:
    pass  # v62 retired, v60 advisor relaunched (research/swap_v60c.log); v62 window = games 214711, 215832
AB_SINCE = "20261106_183551"   # v61 A/B advisor live (research/swap_v61.log); arms before this are the startup label of an old game
since = sys.argv[1] if len(sys.argv) > 1 else None
if since is None:
    # the second (proper) swap time from the swap log, if present
    t0 = None
    try:
        for l in open("C:/liars/research/swap_v60.log", encoding="utf8"):
            if "swap_v60b" in l and "started" in l: t0 = l[:8]
            if "done" in l and t0: since = "20261106_" + t0.replace(":", "")
    except FileNotFoundError:
        pass
if since is None: since = "20261106_040000"
rows = []
for f in sorted(glob.glob("C:/liars/logs/game_*.jsonl")):
    gid = os.path.basename(f)[5:20]; seats = 0; r = None; tiers = []
    for l in open(f, encoding="utf8"):
        if '"e":"mmr"' in l and '"when":"start"' in l and r is None:
            try: r = json.loads(l)
            except ValueError: break
        elif '"e":"seat"' in l:
            seats += 1
            try:
                sj = json.loads(l)
                if not sj.get("me"): tiers.append(sj.get("tier") or 0)
            except ValueError: pass
        elif '"e":"round"' in l: break
    if r: rows.append((gid, r["deck_wins"], r["deck_losses"], r["deck_mmr"], seats, max(tiers) if tiers else 0))
games = wins = 0; mmr_start = None
import collections; by = collections.defaultdict(lambda: [0, 0])
bt = collections.defaultdict(lambda: [0, 0])
arms = {}
try:
    for l in open("C:/liars/research/ab_arms.jsonl", encoding="utf8"):
        try: a = json.loads(l); arms[a["game"]] = a["arm"]
        except ValueError: pass
except FileNotFoundError:
    pass
ba = collections.defaultdict(lambda: [0, 0])
v62 = [0, 0]; v62by = collections.defaultdict(lambda: [0, 0])
v63 = [0, 0]; v63by = collections.defaultdict(lambda: [0, 0])
for (g, w, l, m, s, t), (g2, w2, l2, m2, s2, t2) in zip(rows, rows[1:]):
    if g < since: continue
    if g in EXCLUDE: continue
    if mmr_start is None: mmr_start = m
    dw, dl = w2 - w, l2 - l
    if dw + dl == 1:
        if V63_SINCE and g >= V63_SINCE:
            v63[0] += 1; v63[1] += dw; v63by[s][0] += 1; v63by[s][1] += dw; continue
        if V62_SINCE <= g < V60_RESUME:
            v62[0] += 1; v62[1] += dw; v62by[s][0] += 1; v62by[s][1] += dw; continue
        if arms.get(g) == "data" and g >= AB_SINCE:
            ba["data"][0] += 1; ba["data"][1] += dw; continue   # not part of the v60 baseline
        games += 1; wins += dw; by[s][0] += 1; by[s][1] += dw
        k = "all tier 0" if t == 0 else "max opp tier 1-2" if t <= 2 else "max opp tier 3+"; bt[k][0] += 1; bt[k][1] += dw
        if g in arms and g >= AB_SINCE: ba["net"][0] += 1; ba["net"][1] += dw
pending = rows[-1][0] >= since
print(f"v60 baseline (v60 + net arm; v62 window and v63 trial excluded) since {since}: {games} games decided, {wins} wins, win rate {wins / games if games else float('nan'):.3f}"
      + (f" +/- {1.96 * math.sqrt(wins / games * (1 - wins / games) / games):.3f}" if games else ""))
print("by seats (bot included): " + ", ".join(f"{k}-seat {v[1]}/{v[0]} ({100 * v[1] / v[0]:.0f}%)" for k, v in sorted(by.items())))
print("by opponents: " + ", ".join(f"{k} {v[1]}/{v[0]} ({100 * v[1] / v[0]:.0f}%)" for k, v in sorted(bt.items())))
if v62[0]: print(f"v62 safe-hand guard {V62_SINCE}..{V60_RESUME} (2 games): {v62[1]}/{v62[0]} ({100 * v62[1] / v62[0]:.0f}%)" + (f" +/- {100 * 1.96 * math.sqrt(v62[1] / v62[0] * (1 - v62[1] / v62[0]) / v62[0]):.0f}" if v62[0] else "") + "; by seats " + ", ".join(f"{k}-seat {v[1]}/{v[0]}" for k, v in sorted(v62by.items())))
if V63_SINCE: print(f"v63 truth-first trial since {V63_SINCE}: {v63[1]}/{v63[0]}" + (f" ({100 * v63[1] / v63[0]:.0f}%) +/- {100 * 1.96 * math.sqrt(v63[1] / v63[0] * (1 - v63[1] / v63[0]) / v63[0]):.0f}" if v63[0] else "") + "; by seats " + ", ".join(f"{k}-seat {v[1]}/{v[0]}" for k, v in sorted(v63by.items())) + "; stop rule: < 35% after 60 games")
if ba: print("A/B call arms (v61, paused 21:50): " + ", ".join(f"{k} {v[1]}/{v[0]} ({100 * v[1] / v[0]:.0f}%)" for k, v in sorted(ba.items())))
print(f"bar: > {BAR} over >= 200 games; games still needed: {max(0, 200 - games)}; rating {mmr_start} -> {rows[-1][3]}" + ("; one game in progress" if pending else ""))
