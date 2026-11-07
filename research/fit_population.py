"""Fit the ranked human population from the logs and export it for the simulator.

Outputs
  research/fit_population.json   everything below, for the record
  rust/src/fit_data.rs           constants read by agents::FitHuman

Play model : P(action | real, junk, devil_card, opener, 1v1), 10 actions incl. the Devil play,
             smoothed toward the (real, junk, devil_card) marginal, then toward uniform-legal.
             Per-player bluff latent: log-odds offset ~ N(0, BLUFF_SD) (method of moments).
Call model : logistic on public state + caller's own cards (+ real cards when known) + caller trait.
             Trait = empirical-Bayes shrunk call rate, leave-one-out; population N(TRAIT_MEAN, TRAIT_SD).
Devil mode only (ranked is Devil mode). Grouped-by-game CV for the reported metrics.
"""
import json, collections, sys, math
import numpy as np

PLAYS = "C:/liars/data/plays.jsonl"; REV = "C:/liars/data/reveals.jsonl"
OUT_JSON = "C:/liars/research/fit_population.json"; OUT_RS = "C:/liars/rust/src/fit_data.rs"
ACTS = [None, (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]   # 0 = Devil play
NA = len(ACTS)

def load(p):
    out = []
    for l in open(p, encoding="utf8"):
        try: out.append(json.loads(l))
        except ValueError: pass
    return out

def hand_counts(hand, table):
    real = sum(1 for c in hand if c == table or c == "Joker")
    dev = sum(1 for c in hand if c == "Devil")
    return real, len(hand) - real - dev, dev

def legal(m, nm, d):
    ok = np.zeros(NA, bool)
    ok[0] = d > 0
    for i, a in enumerate(ACTS[1:], 1):
        ok[i] = a[0] <= m and a[1] <= nm
    return ok

def key(m, nm, d, opener, two):
    return (((m * 6 + nm) * 2 + d) * 2 + opener) * 2 + two

plays = [r for r in load(PLAYS) if r.get("mode") == "Devil"]
revs = load(REV)
print("devil-mode play rows", len(plays))

# ---------------- play model ----------------
counts = np.zeros((288, NA)); marg = np.zeros((6, 6, 2, NA))
choice = collections.defaultdict(list)   # player -> bluff indicators where both options existed
era_bluff = collections.defaultdict(list)
rows_play = 0
for r in plays:
    if r.get("play_is_me"): continue
    pl = r["play"]; hand = r.get("claimer_hand"); played = r.get("played_cards") or []
    if not hand or len(hand) != pl["n"] + pl["left"] or len(played) != pl["n"]: continue
    m, nm, d = hand_counts(hand, r["table"])
    if played == ["Devil"]:
        a = 0
    else:
        t = sum(1 for c in played if c == r["table"] or c == "Joker")
        b = len(played) - t
        if "Devil" in played: continue   # Devil thrown inside a multi-card claim: not a simulator action
        try: a = ACTS.index((t, b))
        except ValueError: continue
    opener = 1 if not r["claims_before"] else 0
    two = 1 if r["n_alive"] == 2 else 0
    counts[key(m, nm, d, opener, two), a] += 1
    marg[m, nm, d, a] += 1
    rows_play += 1
    if m >= 1 and nm >= 1 and a != 0:
        bl = 1 if ACTS[a][1] > 0 else 0
        choice[pl["name"]].append(bl)
        era_bluff["ranked" if r["file"] >= "20261103" else "public"].append(bl)
print("play rows used", rows_play)
for e, v in era_bluff.items(): print(f"  bluff rate with a real choice, {e}: {np.mean(v):.3f} (n={len(v)})")

ALPHA = 3.0
table = np.zeros((288, NA), np.float32)
for m in range(6):
    for nm in range(6):
        for d in range(2):
            if m + nm + d > 5 or m + nm + d == 0: continue
            lg = legal(m, nm, d).astype(float)
            uni = lg / lg.sum()
            mc = marg[m, nm, d]
            pm = (mc + ALPHA * uni) / (mc.sum() + ALPHA)
            for op in range(2):
                for two in range(2):
                    k = key(m, nm, d, op, two); c = counts[k]
                    p = (c + ALPHA * pm) / (c.sum() + ALPHA)
                    p = p * lg; p /= p.sum()
                    table[k] = p

# per-player bluff latent (method of moments on players with >= 8 real choices)
rates = [(np.mean(v), len(v)) for v in choice.values() if len(v) >= 8]
p0 = np.mean([x for v in choice.values() for x in v])
vobs = np.var([r for r, _ in rates]); ewithin = np.mean([p0 * (1 - p0) / n for _, n in rates])
vb = max(vobs - ewithin, 1e-4)
bluff_sd_logit = math.sqrt(vb) / (p0 * (1 - p0))
print(f"bluff latent: players {len(rates)}, mean rate {p0:.3f}, sd between {math.sqrt(vb):.3f} (observed {math.sqrt(vobs):.3f}), logit sd {bluff_sd_logit:.2f}")

# ---------------- call model ----------------
called = {(r["file"], r["claim"]["p"], round(r["claim"].get("think", 0), 3), r["round"], len(r["claims"])): r for r in revs}
bygame = collections.defaultdict(list)
for r in plays: bygame[r["file"]].append(r)
revby = collections.defaultdict(list)
for rv in revs: revby[rv["file"]].append(rv)
rows = []
for gid, L in bygame.items():
    L.sort(key=lambda r: r["t"])
    gr = sorted(revby[gid], key=lambda v: v["t"])
    for i, r in enumerate(L):
        pl = r["play"]; cb = r["claims_before"]
        prev = [v for v in gr if v["t"] < r["t"] and v["claim"]["p"] == pl["p"]]
        rec_called = min(len(prev), 8); rec_lies = min(sum(1 for v in prev if v["lie"]), 8)
        rev = called.get((gid, pl["p"], round(pl.get("think", 0), 3), r["round"], len(cb) + 1))
        nxt = L[i + 1] if i + 1 < len(L) and L[i + 1]["round"] == r["round"] else None
        if rev is not None: caller, label = rev["caller"], 1
        elif nxt is not None: caller, label = {"p": nxt["play"]["p"], "name": nxt["play"]["name"]}, 0
        else: continue
        if caller.get("p") is None: continue
        ps = {p["p"]: p for p in r["players"]}
        if caller["p"] not in ps or pl["p"] not in ps: continue
        if ps[caller["p"]].get("me"): continue             # the bot's own call decision (its seat name varies: Gamer / 22 / ...)
        c = ps[caller["p"]]; cl = ps[pl["p"]]
        # A hand-emptying claim is still a free decision when someone besides the claimer and the caller holds cards.
        if pl["left"] == 0 and not any(p["alive"] and p["cards"] > 0 and p["p"] not in (pl["p"], caller["p"]) for p in r["players"]):
            continue                                       # forced call: not a decision
        ah = r.get("all_hands") or {}
        ch = ah.get(caller["name"])
        known = 1 if ch is not None and len(ch) == c["cards"] else 0
        creal = hand_counts(ch, r["table"])[0] if known else 0
        truths = 7 if r["n_alive"] >= 3 else 8
        total_incl = sum(x["n"] for x in cb) + pl["n"]
        rows.append(dict(game=gid, caller=caller["name"], label=label, bot=1 if r.get("play_is_me") else 0,
            n=pl["n"], left=pl["left"], plays_before=min(len(cb), 8), total_before=min(sum(x["n"] for x in cb), 12),
            claimer_earlier=sum(x["n"] for x in cb if x["p"] == pl["p"]), n_alive=r["n_alive"], holders=r["holders"],
            caller_cards=c["cards"], caller_pulls=c["pulls"], claimer_pulls=cl["pulls"], is_opener=1 if not cb else 0,
            round_proxy=min(1 + sum(p["pulls"] for p in r["players"]), 10), known=known, creal=creal * known,
            excess=max(0, total_incl + creal - truths) if known else 0,
            rec_called=rec_called, rec_lies=rec_lies, lie_rate=(rec_lies + 0.5) / (rec_called + 2.0)))
print("free call decisions", len(rows), "called", sum(r["label"] for r in rows), "caller hand known", sum(r["known"] for r in rows))
rows = [r for r in rows if r["known"]]   # knownness is per game (392 all / 113 none); the ranked era is 99.96% known
print("fitting on hand-known rows", len(rows), "games", len({r["game"] for r in rows}))

# caller trait: empirical Bayes, leave-one-out
byc = collections.defaultdict(list)
for r in rows: byc[r["caller"]].append(r["label"])
q0 = np.mean([r["label"] for r in rows])
crates = [(np.mean(v), len(v)) for v in byc.values() if len(v) >= 10]
vobs = np.var([x for x, _ in crates]); ew = np.mean([q0 * (1 - q0) / n for _, n in crates]); vbc = max(vobs - ew, 1e-4)
kappa = q0 * (1 - q0) / vbc
print(f"call trait: callers {len(crates)}, mean {q0:.3f}, sd between {math.sqrt(vbc):.3f} (observed {math.sqrt(vobs):.3f}), kappa {kappa:.1f}")
for r in rows:
    v = byc[r["caller"]]; n = len(v) - 1; k = sum(v) - r["label"]
    r["trait"] = (k + q0 * kappa) / (n + kappa)

FEATS = ["n", "left", "left0", "left2", "plays_before", "total_before", "claimer_earlier", "n_alive", "holders", "caller_cards",
         "caller_pulls", "claimer_pulls", "is_opener", "big", "two", "round_proxy", "creal", "creal0", "excess", "trait", "rec_called", "rec_lies", "lie_rate"]
def vec(r):
    return [r["n"], r["left"], 1 if r["left"] == 0 else 0, 1 if r["left"] <= 2 else 0, r["plays_before"], r["total_before"], r["claimer_earlier"],
            r["n_alive"], r["holders"], r["caller_cards"], r["caller_pulls"], r["claimer_pulls"], r["is_opener"],
            1 if r["n"] >= 3 else 0, 1 if r["n_alive"] == 2 else 0, r["round_proxy"], r["creal"], 1 if r["creal"] == 0 else 0, r["excess"], r["trait"], r["rec_called"], r["rec_lies"], r["lie_rate"]]
X = np.array([vec(r) for r in rows], float); y = np.array([r["label"] for r in rows]); groups = np.array([r["game"] for r in rows])
human = np.array([r["bot"] == 0 for r in rows])
mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
Z = (X - mu) / sd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss
def cv(cols, mask):
    pr = np.zeros(mask.sum()); Zm, ym, gm = Z[mask][:, cols], y[mask], groups[mask]
    for tr, te in GroupKFold(5).split(Zm, ym, gm):
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(Zm[tr], ym[tr]); pr[te] = lr.predict_proba(Zm[te])[:, 1]
    return roc_auc_score(ym, pr), log_loss(ym, pr)
allc = list(range(len(FEATS)))
norec = [i for i, f in enumerate(FEATS) if f not in ("rec_called", "rec_lies", "lie_rate")]
noh = [i for i in norec if FEATS[i] not in ("creal", "creal0", "excess")]
nohnt = [i for i in noh if FEATS[i] != "trait"]
for name, cols in [("state only", nohnt), ("+trait", noh), ("+caller hand", norec), ("+claimer record", allc)]:
    auc, ll = cv(cols, human); print(f"  call model {name:<13} held-out AUC {auc:.3f} log-loss {ll:.4f}")
# Export the model without the claimer's in-game record: it adds nothing held-out (AUC +0.001) and its
# coefficients are collinear; the simulator keeps the 19 state + own-hand + trait features.
EXP = norec
lr = LogisticRegression(C=1.0, max_iter=1000).fit(Z[human][:, EXP], y[human])
W = lr.coef_[0]; B = lr.intercept_[0]
FEATS = [FEATS[i] for i in EXP]; mu = mu[EXP]; sd = sd[EXP]; Z = Z[:, EXP]
print("  coefficients (standardised):", ", ".join(f"{f} {w:+.3f}" for f, w in zip(FEATS, W)))
# bot tax, for the record: residual on the bot's claims
pb = lr.predict_proba(Z[~human])[:, 1]
ob = y[~human].mean(); eb = pb.mean()
print(f"  bot's claims: called {ob:.3f} vs predicted {eb:.3f} (n={int((~human).sum())}) -> log-odds tax {math.log(ob/(1-ob)) - math.log(eb/(1-eb)):+.3f}")

out = dict(acts=ACTS, play_table=table.tolist(), bluff_sd_logit=bluff_sd_logit, bluff_rate=p0,
           call=dict(feats=FEATS, w=W.tolist(), b=float(B), mu=mu.tolist(), sd=sd.tolist()),
           trait_mean=float(q0), trait_sd=float(math.sqrt(vbc)), rows_play=rows_play, rows_call=len(rows))
json.dump(out, open(OUT_JSON, "w"), indent=1)

def arr(v): return "[" + ", ".join(f"{x:.5}" for x in v) + "]"
with open(OUT_RS, "w") as f:
    f.write("// Generated by research/fit_population.py. Do not edit.\n")
    f.write(f"// Devil-mode human population fitted to {rows_play} fully observed plays and {len(rows)} free call decisions.\n")
    f.write(f"pub const FIT_NA: usize = {NA};\n")
    f.write("/// (t, b) per action; action 0 is the Devil play.\n")
    f.write("pub const FIT_ACTS: [(u8, u8); FIT_NA] = [" + ", ".join("(0, 0)" if a is None else f"({a[0]}, {a[1]})" for a in ACTS) + "];\n")
    f.write("/// P(action | real, junk, devil_card, opener, 1v1); index = (((real*6+junk)*2+devil)*2+opener)*2+two.\n")
    f.write("pub const FIT_PLAY: [[f32; FIT_NA]; 288] = [\n")
    for k in range(288): f.write("    " + arr(table[k]) + ",\n")
    f.write("];\n")
    f.write(f"pub const FIT_BLUFF_SD: f64 = {bluff_sd_logit:.4f};\n")
    f.write(f"pub const FIT_TRAIT_MEAN: f64 = {q0:.4f};\npub const FIT_TRAIT_SD: f64 = {math.sqrt(vbc):.4f};\n")
    f.write(f"pub const FIT_CALL_NF: usize = {len(FEATS)};\n")
    f.write("/// Features: " + ", ".join(FEATS) + "\n")
    f.write(f"pub const FIT_CALL_W: [f64; FIT_CALL_NF] = {arr(W)};\n")
    f.write(f"pub const FIT_CALL_B: f64 = {B:.5};\n")
    f.write(f"pub const FIT_CALL_MU: [f64; FIT_CALL_NF] = {arr(mu)};\n")
    f.write(f"pub const FIT_CALL_SD: [f64; FIT_CALL_NF] = {arr(sd)};\n")
print("wrote", OUT_RS)
