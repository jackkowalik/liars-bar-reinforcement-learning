"""Export the labelled human-play dataset for release.
Input: data/plays.jsonl (every claim in every logged game, with was_bluff from post-hoc labelling) and data/reveals.jsonl
(claims that were called, with the revealed cards). Output: data/plays_public.jsonl, data/calls_public.jsonl.
Player names are replaced by stable pseudonyms (salted hash); fields derived from other players' hidden cards are dropped
(all_hands, claimer_hand, played_cards, holders). The bot's own hand is kept (it is the bot's information).
Usage: python export_dataset.py <salt> [out_dir]"""
import json, hashlib, sys, os, collections
salt = sys.argv[1] if len(sys.argv) > 1 else "liars-bar-bot"
out = sys.argv[2] if len(sys.argv) > 2 else "C:/liars/liars-bar-bot/data"
os.makedirs(out, exist_ok=True)
def pid(name):
    if name is None: return None
    return "p" + hashlib.sha1((salt + "|" + str(name)).encode("utf8")).hexdigest()[:10]
def gid(game):
    return "g" + hashlib.sha1((salt + "|" + str(game)).encode("utf8")).hexdigest()[:10]
np = nc = 0; names = set()
with open(f"{out}/plays_public.jsonl", "w", encoding="utf8") as w:
    for l in open("C:/liars/data/plays.jsonl", encoding="utf8"):
        try: r = json.loads(l)
        except ValueError: continue
        game = r["id"].split(":")[0] if ":" in r["id"] else r.get("file", "")[:20]
        play = dict(r["play"]); names.add(play.get("name")); play["player"] = pid(play.pop("name", None)); play.pop("slots", None)
        for e in play.get("earlier", []) or []:
            if isinstance(e, dict) and "name" in e: e["player"] = pid(e.pop("name"))
        cb = []
        for c in r.get("claims_before") or []:
            c = dict(c) if isinstance(c, dict) else {"v": c}
            if "name" in c: c["player"] = pid(c.pop("name"))
            cb.append(c)
        o = dict(game=gid(game), t=r.get("t"), mode=r.get("mode"), ranked=r.get("ranked"), seats=r.get("game_players"),
                 round=r.get("round"), table=r.get("table"), players=[pid(x) for x in (r.get("players") or [])],
                 n_alive=r.get("n_alive"), claims_before=cb, play=play, play_is_bot=r.get("play_is_me"),
                 bot_hand=r.get("my_hand") if r.get("play_is_me") is not None else None, bot_matching=r.get("my_real"),
                 was_bluff=r.get("was_bluff"), label_source=r.get("label_source"))
        w.write(json.dumps(o, ensure_ascii=False) + "\n"); np += 1
with open(f"{out}/calls_public.jsonl", "w", encoding="utf8") as w:
    for l in open("C:/liars/data/reveals.jsonl", encoding="utf8"):
        try: r = json.loads(l)
        except ValueError: continue
        game = r["id"].split(":")[0] if ":" in r["id"] else r.get("file", "")[:20]
        claim = dict(r["claim"]) if isinstance(r.get("claim"), dict) else r.get("claim")
        if isinstance(claim, dict) and "name" in claim: claim["player"] = pid(claim.pop("name"))
        claims = []
        for c in r.get("claims") or []:
            c = dict(c) if isinstance(c, dict) else {"v": c}
            if "name" in c: c["player"] = pid(c.pop("name"))
            claims.append(c)
        o = dict(game=gid(game), t=r.get("t"), mode=r.get("mode"), ranked=r.get("ranked"), seats=r.get("game_players"),
                 round=r.get("round"), table=r.get("table"), players=[pid(x) for x in (r.get("players") or [])],
                 n_alive=r.get("n_alive"), claims=claims, claim=claim, caller=pid(r.get("caller")),
                 caller_is_bot=r.get("caller_is_me"), claimer_is_bot=r.get("claimer_is_me"), lie=r.get("lie"),
                 revealed=r.get("revealed"), pulled=r.get("pulled"), devil=r.get("devil"), bot_hand=r.get("my_hand"),
                 bot_matching=r.get("my_real"))
        w.write(json.dumps(o, ensure_ascii=False) + "\n"); nc += 1
print(f"plays_public.jsonl: {np} rows; calls_public.jsonl: {nc} rows; distinct players pseudonymised: {len(names)}")
