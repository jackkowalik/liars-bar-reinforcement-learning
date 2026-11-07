"""Export the labelled human-play dataset for release.
Input: data/plays.jsonl (every claim in every logged game, with was_bluff from post-hoc labelling) and data/reveals.jsonl
(claims that were called, with the revealed cards). Output: data/plays_public.jsonl (labelled claims with hands), data/plays_unlabelled.jsonl (early games without
snapshots), data/calls_public.jsonl.
Player names are replaced by stable pseudonyms (salted hash); fields are pseudonymised consistently across files. Hidden hands (every seat's cards at the time of the claim, the claimer's hand
and the cards actually thrown) are included where known: they are the ground truth behind the labels.
Usage: python export_dataset.py <salt> [out_dir]"""
import json, hashlib, sys, os, collections
salt = sys.argv[1] if len(sys.argv) > 1 else "liars-bar-bot"
out = sys.argv[2] if len(sys.argv) > 2 else "C:/liars/liars-bar-bot/data"
os.makedirs(out, exist_ok=True)
def pid(name):
    if name is None: return None
    return "p" + hashlib.sha1((salt + "|" + str(name)).encode("utf8")).hexdigest()[:10]
def seats_of(players):
    out = []
    for x in players or []:
        if isinstance(x, dict):
            out.append(dict(seat=x.get("p"), player=pid(x.get("name")), alive=x.get("alive"), cards=x.get("cards"), pulls=x.get("pulls"),
                            is_bot=x.get("me"), tier=x.get("rank"), level=x.get("level")))
        else:
            out.append(dict(player=pid(x)))
    return out
def hands_of(ah):
    if not ah: return None
    return {pid(name): hand for name, hand in ah.items()}
def gid(game):
    return "g" + hashlib.sha1((salt + "|" + str(game)).encode("utf8")).hexdigest()[:10]
np = nc = 0; names = set()
wl = open(f"{out}/plays_public.jsonl", "w", encoding="utf8"); wu = open(f"{out}/plays_unlabelled.jsonl", "w", encoding="utf8")
with wl, wu:
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
                 round=r.get("round"), table=r.get("table"), players=seats_of(r.get("players")),
                 n_alive=r.get("n_alive"), claims_before=cb, play=play, play_is_bot=r.get("play_is_me"),
                 claimer_hand=r.get("claimer_hand"), thrown=r.get("played_cards"), hands=hands_of(r.get("all_hands")),
                 bot_hand=r.get("my_hand") if r.get("play_is_me") is not None else None, bot_matching=r.get("my_real"),
                 was_bluff=r.get("was_bluff"), label_source=r.get("label_source"))
        (wl if o["was_bluff"] is not None else wu).write(json.dumps(o, ensure_ascii=False) + "\n"); np += 1
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
                 round=r.get("round"), table=r.get("table"), players=seats_of(r.get("players")),
                 n_alive=r.get("n_alive"), claims=claims, claim=claim, caller=pid(r.get("caller")), hands=hands_of(r.get("all_hands")),
                 caller_is_bot=r.get("caller_is_me"), claimer_is_bot=r.get("claimer_is_me"), lie=r.get("lie"),
                 revealed=r.get("revealed"), pulled=r.get("pulled"), devil=r.get("devil"), bot_hand=r.get("my_hand"),
                 bot_matching=r.get("my_real"))
        w.write(json.dumps(o, ensure_ascii=False) + "\n"); nc += 1
print(f"plays_public.jsonl: {np} rows; calls_public.jsonl: {nc} rows; distinct players pseudonymised: {len(names)}")
