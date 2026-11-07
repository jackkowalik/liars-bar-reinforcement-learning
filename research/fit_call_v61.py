"""Fit the v61 data call model on every labelled free call spot and export it for the advisor.

Rows are built exactly as in call_scorer.py (same features, same definitions) so there is no train/serve skew:
the advisor (watch.rs, LIARS_CALL_RULE=data) recomputes the same 18 features live.
Output: research/call_model_v61.json = {features, w, b, mu, sd, p0, K, priors: {name: [n_labelled, n_lies]}, thresh}
Also prints the time-split check (fit < Nov 5, test >= Nov 5) for the exported feature set.
"""
import json, collections, math, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
sys.argv = [sys.argv[0]]
src = open("C:/liars/research/call_scorer.py", encoding="utf8").read()
# reuse the row construction and prior code verbatim (everything up to the model definitions)
head = src.split("y = np.array([r[\"lie\"] for r in rows])")[0]
ns = {}
exec(compile(head, "call_scorer_rows", "exec"), ns)
rows, by_cl, p0, K = ns["rows"], ns["by_cl"], ns["p0"], ns["K"]
FEATS = ["n", "left", "left0", "plays_before", "total_before", "claimer_earlier", "n_alive", "holders", "think",
         "my_cards", "my_real", "my_real0", "my_devil", "impossible", "p_true", "prior", "rec_called", "rec_lies"]
X = np.array([[r[f] for f in FEATS] for r in rows], float); y = np.array([r["lie"] for r in rows])
mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1; Z = (X - mu) / sd
# time-split check first
tr = np.array([r["game"] < "20261105" for r in rows]); te = ~tr
lr = LogisticRegression(C=1.0, max_iter=3000).fit(Z[tr], y[tr]); pt = lr.predict_proba(Z[te])[:, 1]
Gt = len({r["game"] for r in rows if r["game"] >= "20261105"})
print(f"time split: {tr.sum()} fit / {te.sum()} test spots ({Gt} games): AUC {roc_auc_score(y[te], pt):.3f}")
for t in (0.5, 0.55, 0.6, 0.65, 0.7):
    m = pt >= t; lies = (m & (y[te] == 1)).sum(); calls = m.sum()
    print(f"  thresh {t}: calls/game {calls / Gt:.2f} precision {lies / max(calls, 1):.2f} net pulls/game {(2 * lies - calls) / Gt:+.2f}")
# final fit on everything
lr = LogisticRegression(C=1.0, max_iter=3000).fit(Z, y)
print("coefficients (standardized):")
for f, w in sorted(zip(FEATS, lr.coef_[0]), key=lambda x: -abs(x[1])): print(f"  {f:<16} {w:+.3f}")
# per-player prior table from ALL labelled plays (live use: the claimer's history from other games)
priors = {}
for name, d in by_cl.items():
    n = sum(v[0] for v in d.values()); k = sum(v[1] for v in d.values())
    if name and n > 0: priors[name] = [n, k]
thresh = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
out = dict(features=FEATS, w=[float(x) for x in lr.coef_[0]], b=float(lr.intercept_[0]), mu=[float(x) for x in mu],
           sd=[float(x) for x in sd], p0=p0, K=K, thresh=thresh, priors=priors, n_rows=len(rows),
           note="features as in call_scorer.py; prior=(k+p0*K)/(n+K) from priors[name] (p0 if unknown); call iff P(lie)>=thresh")
json.dump(out, open("C:/liars/research/call_model_v61.json", "w"), indent=1)
print(f"exported call_model_v61.json: {len(rows)} rows, {len(priors)} player priors, thresh {thresh}")
