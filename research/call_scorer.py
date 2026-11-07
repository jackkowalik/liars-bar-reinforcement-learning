"""Offline scoring of call policies on real human claims the bot had to answer.

Every opponent claim where the bot was the next actor with a free decision and the claimer's hand is known (so the
truth of the claim is known). A call policy is scored by what it would have done on these spots:
  calls/game, precision (share of called claims that were lies), lies caught/game, truths called/game,
  net pulls/game = lies caught - truths called (a call on a lie makes the claimer pull, a call on a truth makes me pull).
Policies: never call; card count says impossible; logistic lie models with grouped-by-game CV (state only, + my hand,
+ claimer's bluff prior from OTHER games, + claimer's in-game record) at several call thresholds; and what the bot
actually did, by era.
"""
import json, collections, math, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

def load(p):
    out = []
    for l in open(p, encoding="utf8"):
        try: out.append(json.loads(l))
        except ValueError: pass
    return out

plays = [r for r in load("C:/liars/data/plays.jsonl") if r.get("mode") == "Devil"]
revs = load("C:/liars/data/reveals.jsonl")
called = {(r["file"], r["claim"]["p"], round(r["claim"].get("think", 0), 3), r["round"], len(r["claims"])): r for r in revs}
revby = collections.defaultdict(list)
for v in revs: revby[v["file"]].append(v)
bygame = collections.defaultdict(list)
for r in plays: bygame[r["file"]].append(r)

def real_count(hand, table):
    return sum(1 for c in hand if c == table or c == "Joker"), sum(1 for c in hand if c == "Devil")

def comb(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0

def p_true(n_claim, my_real, others_cards_total, total_claimed_before, truths):
    """Chance the claim is true by card count: hypergeometric on the cards I cannot see, treating earlier claims
    this round as true (the advisor's `pt`)."""
    unseen = max(others_cards_total, 1)
    reals_unseen = max(truths - my_real - total_claimed_before, 0)
    if n_claim > reals_unseen: return 0.0
    # P(at least n_claim real among the claimer's n_claim cards drawn from unseen)
    return comb(reals_unseen, n_claim) * comb(unseen - reals_unseen, 0) / max(comb(unseen, n_claim), 1)

rows = []
for gid, L in bygame.items():
    L.sort(key=lambda r: r["t"])
    gr = sorted(revby[gid], key=lambda v: v["t"])
    for i, r in enumerate(L):
        if r.get("play_is_me"): continue
        pl = r["play"]; cb = r["claims_before"]
        hand = r.get("claimer_hand")
        if not hand or r.get("was_bluff") is None or len(hand) != pl["n"] + pl["left"]: continue
        rev = called.get((gid, pl["p"], round(pl.get("think", 0), 3), r["round"], len(cb) + 1))
        nxt = L[i + 1] if i + 1 < len(L) and L[i + 1]["round"] == r["round"] else None
        if rev is not None: caller_is_me = bool(rev.get("caller_is_me")); i_called = 1 if caller_is_me else 0
        elif nxt is not None: caller_is_me = bool(nxt.get("play_is_me")); i_called = 0
        else: continue
        if not caller_is_me: continue
        me = [p for p in r["players"] if p["me"]]
        if not me or me[0]["cards"] == 0: continue
        me = me[0]
        others_hold = [p for p in r["players"] if p["alive"] and p["cards"] > 0 and not p["me"] and p["p"] != pl["p"]]
        if pl["left"] == 0 and not others_hold: continue          # forced call
        my_hand = r.get("my_hand") or []
        my_real, my_devil = real_count(my_hand, r["table"])
        truths = 7 if r["n_alive"] >= 3 else 8
        total_before = sum(x["n"] for x in cb)
        others_cards = sum(p["cards"] for p in r["players"] if p["alive"] and not p["me"]) + pl["n"]  # before the throw
        prev = [v for v in gr if v["t"] < r["t"] and v["claim"]["p"] == pl["p"]]
        rows.append(dict(game=gid, round=r["round"], cb_len=len(cb), claimer=pl["name"], lie=int(r["was_bluff"]), i_called=i_called,
            n=pl["n"], left=pl["left"], left0=int(pl["left"] == 0), plays_before=min(len(cb), 8), total_before=min(total_before, 12),
            claimer_earlier=sum(x["n"] for x in cb if x["p"] == pl["p"]), n_alive=r["n_alive"], holders=r["holders"],
            my_cards=me["cards"], my_real=my_real, my_real0=int(my_real == 0), my_devil=my_devil,
            impossible=int(total_before + pl["n"] + my_real > truths),
            p_true=p_true(pl["n"], my_real, others_cards, total_before, truths),
            rec_called=min(len(prev), 8), rec_lies=min(sum(1 for v in prev if v["lie"]), 8),
            think=min(pl.get("think", 0) or 0, 30), era="net" if gid < "20261103" else "rules" if gid < "20261106_0340" else "v60"))
G = len({r["game"] for r in rows})
print(f"bot's free call decisions on labelled human claims: {len(rows)} in {G} games; lie share {np.mean([r['lie'] for r in rows]):.3f}")

# claimer bluff prior from OTHER games (leave-one-game-out), shrunk toward the population rate
by_cl = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
for r in plays:
    if r.get("play_is_me") or r.get("was_bluff") is None or not r.get("claimer_hand"): continue
    d = by_cl[r["play"]["name"]][r["file"]]; d[0] += 1; d[1] += int(r["was_bluff"])
p0 = 0.43; K = 10.0
for r in rows:
    d = by_cl[r["claimer"]]
    n = sum(v[0] for g, v in d.items() if g != r["game"]); k = sum(v[1] for g, v in d.items() if g != r["game"])
    r["prior"] = (k + p0 * K) / (n + K); r["prior_n"] = n

y = np.array([r["lie"] for r in rows]); groups = np.array([r["game"] for r in rows]); ic = np.array([r["i_called"] for r in rows])
era = np.array([r["era"] for r in rows])
FE = {
    "state": ["n", "left", "left0", "plays_before", "total_before", "claimer_earlier", "n_alive", "holders", "think"],
    "+my hand": ["my_cards", "my_real", "my_real0", "my_devil", "impossible", "p_true"],
    "+claimer prior (other games)": ["prior"],
    "+in-game record": ["rec_called", "rec_lies"],
}
def mat(cols): return np.array([[r[c] for c in cols] for r in rows], float)
def cv_probs(cols):
    X = mat(cols); mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1; Z = (X - mu) / sd
    pr = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(Z, y, groups):
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Z[tr], y[tr]); pr[te] = lr.predict_proba(Z[te])[:, 1]
    return pr

def score(call_mask, name):
    calls = call_mask.sum(); lies = (call_mask & (y == 1)).sum(); truths = calls - lies
    print(f"  {name:<44} calls/game {calls / G:5.2f}  precision {lies / max(calls, 1):.2f}  lies caught/game {lies / G:.2f}  truths called/game {truths / G:.2f}  net pulls/game {(lies - truths) / G:+.2f}")

print("\npolicies (all eras pooled):")
score(np.zeros(len(y), bool), "never call")
score(np.array([r["impossible"] for r in rows]) == 1, "card count: claim impossible")
score(np.array([r["p_true"] for r in rows]) < 0.2, "card count: claim < 20% likely true")
for e in ("net", "rules", "v60"):
    m = era == e
    if m.sum():
        calls = (ic[m] == 1).sum(); lies = ((ic[m] == 1) & (y[m] == 1)).sum(); g = len({r["game"] for r in rows if r["era"] == e})
        print(f"  {'bot actually called, era ' + e:<44} calls/game {calls / g:5.2f}  precision {lies / max(calls, 1):.2f}  lies caught/game {lies / g:.2f}  truths called/game {(calls - lies) / g:.2f}  net pulls/game {(2 * lies - calls) / g:+.2f}   (n={m.sum()}, {g} games)")

print("\nlie models, grouped-by-game CV:")
cols = []
for name, fs in FE.items():
    cols = cols + fs
    pr = cv_probs(cols)
    print(f" {name:<30} AUC {roc_auc_score(y, pr):.3f}")
    for t in (0.5, 0.6, 0.7, 0.8):
        score(pr >= t, f"    call if P(lie) >= {t}")
# how much the per-player prior adds when there is real history on the claimer
pr_all = cv_probs(sum(FE.values(), [])); pr_noprior = cv_probs(FE["state"] + FE["+my hand"] + FE["+in-game record"])
hist = np.array([r["prior_n"] for r in rows])
for lo, hi in ((0, 0), (1, 9), (10, 29), (30, 10**6)):
    m = (hist >= lo) & (hist <= hi)
    if m.sum() > 50:
        print(f"  claimer with {lo}-{hi if hi < 10**6 else '+'} labelled plays in other games: n={m.sum():5d}  AUC without prior {roc_auc_score(y[m], pr_noprior[m]):.3f}  with prior {roc_auc_score(y[m], pr_all[m]):.3f}")

# ---- the v60 network's own call head, scored on the same spots (research/net_call_replay.py -> net_calls.json) ----
import os
if os.path.exists("C:/liars/research/net_calls.json"):
    nc = {(d["game"], d["round"], d["claimer"], d["idx"]): d for d in json.load(open("C:/liars/research/net_calls.json"))}
    pc = np.full(len(y), np.nan); tp = np.full(len(y), np.nan)
    for i, r in enumerate(rows):
        d = nc.get((r["game"], r["round"], r["claimer"], r["cb_len"] + 1))   # the replay counts the claim being judged
        if d: pc[i] = d["call_pct"] / 100.0; tp[i] = (d["top_play_pct"] or 0) / 100.0
    have = ~np.isnan(pc)
    print(f"\nnetwork call head (v60 settings replay): matched {have.sum()} of {len(y)} spots; AUC {roc_auc_score(y[have], pc[have]):.3f}")
    Gh = len({rows[i]['game'] for i in np.nonzero(have)[0]})
    def score_h(mask, name):
        mask = mask & have
        calls = mask.sum(); lies = (mask & (y == 1)).sum(); truths = calls - lies
        print(f"  {name:<44} calls/game {calls / Gh:5.2f}  precision {lies / max(calls, 1):.2f}  lies caught/game {lies / Gh:.2f}  truths called/game {truths / Gh:.2f}  net pulls/game {(lies - truths) / Gh:+.2f}")
    score_h(np.nan_to_num(pc) > np.nan_to_num(tp), "v60 live rule: call if CALL% > top PLAY%")
    for t in (0.35, 0.5, 0.6, 0.7):
        score_h(np.nan_to_num(pc) >= t, f"network call if P >= {t}")
    pm = cv_probs(sum(FE.values(), []))
    for t in (0.5, 0.6):
        score_h(pm >= t, f"data lie model (+prior) >= {t}, same spots")

# ---- time split: fit on games before Nov 5, test on Nov 5-6 (no look-ahead at all) ----
cols = sum(FE.values(), [])
X = mat(cols); mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1; Z = (X - mu) / sd
tr = np.array([r["game"] < "20261105" for r in rows]); te = ~tr
lr = LogisticRegression(C=1.0, max_iter=2000).fit(Z[tr], y[tr]); pt = lr.predict_proba(Z[te])[:, 1]
Gt = len({r["game"] for r in rows if r["game"] >= "20261105"})
print(f"\ntime split: fit on {tr.sum()} spots before Nov 5, test on {te.sum()} spots in {Gt} games from Nov 5: AUC {roc_auc_score(y[te], pt):.3f}")
for t in (0.5, 0.6, 0.7):
    m = pt >= t; lies = (m & (y[te] == 1)).sum(); calls = m.sum()
    print(f"    call if P(lie) >= {t}: calls/game {calls / Gt:.2f}  precision {lies / max(calls, 1):.2f}  net pulls/game {(2 * lies - calls) / Gt:+.2f}")
mh = te & have
print(f"    network call head on the same test spots (n={mh.sum()}): AUC {roc_auc_score(y[mh], pc[mh]):.3f}; v60 rule net pulls/game {((2 * ((pc > tp) & mh & (y == 1)).sum()) - ((pc > tp) & mh).sum()) / Gt:+.2f}")
