"""Where do the bot's calls go wrong? Every call the bot made since SINCE, with the spot: players alive, claim index in
the round, cards claimed, claimer's cards left after, round total claimed so far, bot's pulls before. Precision by bucket."""
import json, glob, os, collections, sys
SINCE = sys.argv[1] if len(sys.argv) > 1 else "20261104"
calls = []
for f in sorted(glob.glob("C:/liars/logs/game_*.jsonl")):
    gid = os.path.basename(f)[5:20]
    if gid < SINCE: continue
    me = None; idx = 0; total = 0; alive = None; pulls = None; last = None; n_alive = None
    for line in open(f, encoding="utf8"):
        if '"e":"pose"' in line or '"e":"look"' in line or '"e":"slots"' in line or '"e":"anchor"' in line or '"e":"hand"' in line: continue
        try: r = json.loads(line)
        except ValueError: continue
        e = r["e"]
        if e == "seat" and r.get("me"): me = r["p"]
        elif e == "round":
            idx = 0; total = 0; alive = r.get("alive"); pulls = r.get("pulls"); n_alive = sum(alive) if alive else None; last = None
        elif e == "play":
            idx += 1; total += r["n"]; last = r
        elif e == "pull" and r["caller"] == me and last is not None:
            calls.append(dict(game=gid, lie=r["lie"], died=r["died"] and r["p"] == me, n_alive=n_alive, idx=idx, n=last["n"],
                              left=last.get("left"), total=total, my_pulls=(pulls[me] if pulls and me is not None and me < len(pulls) else None),
                              think=last.get("think")))
print(f"bot calls since {SINCE}: {len(calls)}; correct {sum(c['lie'] for c in calls)} ({sum(c['lie'] for c in calls)/len(calls):.0%}); bot died on wrong calls: {sum(c['died'] for c in calls)}")
def tab(name, keyf):
    t = collections.defaultdict(lambda: [0, 0, 0])
    for c in calls:
        k = keyf(c); t[k][0] += 1; t[k][1] += c["lie"]; t[k][2] += c["died"]
    print(f"\n{name}: n calls, correct%, bot deaths")
    for k in sorted(t, key=lambda x: str(x)): n, ok, d = t[k]; print(f"  {str(k):28s} {n:4d}  {ok/n:5.0%}  {d}")
tab("players alive", lambda c: c["n_alive"])
tab("claim index in round", lambda c: min(c["idx"], 6))
tab("cards claimed", lambda c: c["n"])
tab("claimer emptied hand (left==0)", lambda c: c["left"] == 0)
tab("round total claimed (incl. this)", lambda c: "<=4" if c["total"] <= 4 else "5-8" if c["total"] <= 8 else ">8 (impossible all true)")
tab("opening claim of the round, by players alive", lambda c: ("opening" if c["idx"] == 1 else "later", c["n_alive"]))
tab("my pulls before the call", lambda c: c["my_pulls"])
json.dump(calls, open(sys.argv[0].rsplit("/",1)[0] + "/bot_calls.json", "w"))
