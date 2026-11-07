"""Step 1: price the bot's bluffs by the next player's call rate.
From raw game logs since 2026-11-01 (ranked era, seat tiers logged).
- call rate per player name and per tier: calls / opportunities (a turn that follows another player's claim)
- for every bot throw: kind (truth / needless bluff = non-matching thrown while a matching card was in hand / forced bluff),
  next player's tier and call rate, whether it was called, and whether the bot died from it
"""
import json, glob, os, collections, sys
SINCE = sys.argv[1] if len(sys.argv) > 1 else "20261101"
opp = collections.Counter(); calls = collections.Counter()          # by name
topp = collections.Counter(); tcalls = collections.Counter()        # by tier
throws = []   # bot throws
deaths = collections.Counter()
for f in sorted(glob.glob("C:/liars/logs/game_*.jsonl")):
    gid = os.path.basename(f)[5:20]
    if gid < SINCE: continue
    seats = {}; tier = {}; me = None; table = None; hand = None; n = 0
    pending = None   # (claimer, n, kind, next_player) awaiting a call decision
    alive = None
    for line in open(f, encoding="utf8"):
        if '"e":"pose"' in line or '"e":"look"' in line or '"e":"slots"' in line or '"e":"anchor"' in line: continue
        try: r = json.loads(line)
        except ValueError: continue
        e = r["e"]
        if e == "seat":
            seats[r["p"]] = r.get("name"); tier[r["p"]] = r.get("tier") or 0
            if r.get("me"): me = r["p"]
            n = max(n, r["p"] + 1)
        elif e == "round":
            table = r["table"]; hand = r["hand"]; alive = r.get("alive"); pending = None
        elif e == "turn":
            p = r["p"]
            if p == me and "hand" in r: hand = r["hand"]
            if pending is not None and pending["claimer"] != p:
                # this player can call the pending claim
                pending["next"] = p; pending["next_tier"] = tier.get(p, 0); pending["next_name"] = seats.get(p)
                opp[seats.get(p)] += 1; topp[tier.get(p, 0)] += 1
                pending["decided"] = False
        elif e == "play":
            p = r["p"]; k = r["n"]
            if pending is not None and pending.get("next") == p and not pending.get("decided"):
                pending["decided"] = True; pending["called"] = False
                if pending["claimer"] == me: throws.append(pending)
            kind = None
            if p == me and hand is not None and "hand" in r:
                before = list(hand); after = list(r["hand"])
                for c in after:
                    if c in before: before.remove(c)
                thrown = before
                valid = lambda c: c == table or c == 4
                had_valid = sum(1 for c in hand if valid(c))
                if thrown:
                    if all(valid(c) for c in thrown): kind = "truth"
                    elif had_valid >= len(thrown): kind = "needless_bluff"
                    elif had_valid > 0: kind = "partial_bluff"
                    else: kind = "forced_bluff"
                hand = after
            pending = {"game": gid, "claimer": p, "n": k, "left": r.get("left"), "kind": kind, "think": r.get("think"), "table": table}
        elif e == "pull":
            caller = r["caller"]; accused = r["accused"]
            if pending is not None and pending.get("next") == caller and not pending.get("decided"):
                pending["decided"] = True; pending["called"] = True; pending["lie"] = r["lie"]; pending["died"] = r["died"]; pending["puller"] = r["p"]
                calls[seats.get(caller)] += 1; tcalls[tier.get(caller, 0)] += 1
                if pending["claimer"] == me: throws.append(pending)
            if r["died"] and r["p"] == me:
                if caller == me: deaths["wrong call" if not r["lie"] else "right call??"] += 1
                else: deaths["caught: " + str(pending["kind"] if pending and pending["claimer"] == me else "?")] += 1
            pending = None
print(f"since {SINCE}")
print("\ncall rate by tier of the player deciding:")
for t in sorted(topp): print(f"  tier {t}: {tcalls[t]}/{topp[t]} = {tcalls[t]/topp[t]:.1%}")
print("\nbot throws by kind x next player's tier: n, called%, and of those called -> bot pulled (bluff) / caller pulled (truth)")
by = collections.defaultdict(lambda: [0, 0, 0])
for th in throws:
    tb = "t3+" if th["next_tier"] >= 3 else ("t1-2" if th["next_tier"] >= 1 else "t0")
    k = (th["kind"], tb); by[k][0] += 1
    if th.get("called"): by[k][1] += 1; by[k][2] += th.get("died", False)
for k in sorted(by, key=lambda x: (str(x[0]), x[1])):
    n_, c, d = by[k]; print(f"  {str(k[0]):15s} next {k[1]:5s}: {n_:5d} throws, called {c/n_:5.1%}, bot died on {d}")
print("\nbot deaths by cause:", dict(deaths))
# prize estimate: needless bluffs that were called -> the bot pulled; had it thrown the real card, the caller pulls instead
nb_called = [th for th in throws if th["kind"] == "needless_bluff" and th.get("called")]
print(f"\nneedless bluffs called: {len(nb_called)}; bot deaths among them: {sum(1 for th in nb_called if th.get('died'))}")
print("  by next tier:", collections.Counter("t3+" if th["next_tier"] >= 3 else "t<3" for th in nb_called))
json.dump(throws, open("bot_throws.json", "w"))
# per-name call rates for frequent callers
print("\nmost frequent callers (>=40 opportunities):")
for nm, o in sorted(opp.items(), key=lambda x: -x[1])[:12]:
    if o >= 40: print(f"  {str(nm).encode('ascii','replace').decode():20s} {calls[nm]}/{o} = {calls[nm]/o:.0%}")
