"""Fits the simulated human-like players to real lobby behaviour.

Real statistics come from nn/real_stats.py (public log events only); simulated ones from
`liars simstats` with the correction knobs in LIARS_HUMAN_FIT (rust/src/agents.rs human_fit).
The fit minimises a chi-square over call rates, lie-when-called rates and claim sizes in both
deck modes, on all games but the newest `--holdout`, and reports the held-out fit before/after.
"""
import argparse
import glob
import math
import os
import re
import subprocess

import real_stats

EXE = "C:/liars/rust/target-simstats/release/liars.exe"
RANKED = False
LIE_W = 1.0
THREADS = 8
NAMES = ["all calls", "late calls", "opening calls", "late bluffs", "early bluffs", "big-claim calls", "1v1 calls", "over-claimed calls", "whole-hand bluffs", "no-real-cards calls", "model call odds", "own-hand doubt", "no-real-cards (model)"]
LINE = re.compile(r"^\s+(.+?)\s+call rate ([\d.na]+) \(n=\s*(\d+)\)\s+lie when called ([\d.na]+) \(n=\s*(\d+)\)")


def parse(text):
    out = {}
    for ln in text.splitlines():
        m = LINE.match(ln)
        if m:
            out[m[1]] = (float(m[2]), int(m[3]), float(m[4]), int(m[5]))
        elif ln.startswith("claim size"):
            v = [float(x) for x in re.findall(r"\d: ([\d.]+)", ln)]
            out["size"] = v
    return out


def real_table(rows):
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        real_stats.report(rows)
    t = parse(buf.getvalue())
    t["_n"] = len(rows)
    return t


def sim_table(fit, devil, games, seed_net):
    v = ",".join(f"{x:.4f}" for x in fit)
    env = dict(os.environ, LIARS_HUMAN_FIT_RANKED=v, LIARS_ALL_RANKED="1") if RANKED and devil else         dict(os.environ, **{"LIARS_HUMAN_FIT_DEVIL" if devil else "LIARS_HUMAN_FIT": v})
    cmd = [EXE, "simstats", "--games", str(games), "--threads", str(THREADS), "--net", seed_net] + (["--devil"] if devil else [])
    return parse(subprocess.run(cmd, env=env, capture_output=True, text=True).stdout)


def chi2(real, sim):
    s = 0.0
    for k, v in real.items():
        if k.startswith("_") or k == "size" or k not in sim:
            continue
        c, nc, l, nl = v
        sc, _, sl, _ = sim[k]
        for r, x, n, w in ((c, sc, nc, 1.0), (l, sl, nl, LIE_W)):
            if n >= 10 and not math.isnan(r):
                p = min(max(r, 0.05), 0.95)
                s += w * (x - r) ** 2 / (p * (1 - p) / n)
    n = real["_n"]
    for r, x in zip(real["size"], sim["size"]):
        p = min(max(r, 0.05), 0.95)
        s += (x - r) ** 2 / (p * (1 - p) / n)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=int, default=20)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--start", default="1,1,1,1,1,1,1,1,1")
    ap.add_argument("--modes", default="Basic,Devil", help="fit only these modes (Devil has its own fit)")
    ap.add_argument("--since", default="", help="only logs from this time on, e.g. 20261031_07 (ranked)")
    ap.add_argument("--lie-weight", type=float, default=1.0, help="weight on lie-when-called terms")
    ap.add_argument("--ranked", action="store_true", help="fit the ranked Devil players (HUMAN_FIT_RANKED)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", default="", help="comma-separated knob indices to search (default all)")
    args = ap.parse_args()
    global RANKED, LIE_W, THREADS
    RANKED = args.ranked
    LIE_W = args.lie_weight
    THREADS = args.threads
    only = {int(x) for x in args.only.split(",") if x.strip()}
    modes = args.modes.split(",")
    files = sorted(os.path.basename(f) for f in glob.glob("C:/liars/logs/game_*.jsonl")
                   if os.path.basename(f)[5:] >= args.since)
    held = files[-args.holdout:]
    fit_files = files[:-args.holdout]
    real = {m: real_table(real_stats.load(m, only=fit_files)) for m in modes}
    test = {m: real_table(real_stats.load(m, only=held)) for m in modes}
    nets = {"Basic": "C:/liars/nn/basic_live", "Devil": "C:/liars/nn/devil_live"}

    def score(fit, tables, games=args.games):
        return sum(chi2(tables[m], sim_table(fit, m == "Devil", games, nets[m])) for m in tables)

    fit = [float(x) for x in args.start.split(",")]
    best = score(fit, real)
    print(f"start: fit chi2 {best:.1f}  | held-out chi2 {score(fit, test, 8000):.1f}", flush=True)
    steps = [0.5, 0.35, 0.25, 0.18, 0.12, 0.08]
    for rnd in range(args.rounds):
        step = steps[min(rnd, len(steps) - 1)]
        for i in range(len(fit)):
            if only and i not in only:
                continue
            for d in (math.exp(step), math.exp(-step)):
                trial = list(fit)
                trial[i] = min(max(trial[i] * d, 0.03), 6.0)
                s = score(trial, real)
                if s < best:
                    best, fit = s, trial
                    print(f"  round {rnd + 1}: {NAMES[i]} x{fit[i]:.2f} -> chi2 {best:.1f}", flush=True)
                    break
    print("fit: " + ", ".join(f"{n} {x:.2f}" for n, x in zip(NAMES, fit)))
    print("LIARS_HUMAN_FIT=" + ",".join(f"{x:.3f}" for x in fit))
    print(f"held-out chi2: before {score([1.0] * 9 + [0.0, 1.0, 2.0, 1.0], test, 8000):.1f}, after {score(fit, test, 8000):.1f}")
    for m in modes:
        print(f"\n{m} (held-out real vs fitted sim):")
        s = sim_table(fit, m == "Devil", 8000, nets[m])
        for k, v in test[m].items():
            if k.startswith("_") or k == "size":
                continue
            print(f"  {k:<9} call {v[0]:.2f} vs {s[k][0]:.2f}   lie {v[2]:.2f} vs {s[k][2]:.2f}  (real n {v[1]}/{v[3]})")


if __name__ == "__main__":
    main()
