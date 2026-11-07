"""Self-play improvement (PPO) for the transformer network, same setup as ppo.py:
log-fitted human-like players (a third heavy callers), champion copies, honest and probability
bots, frozen past versions (league); reward = change in win probability over the round;
KL leash to the starting network; saved only if no opponent type gets worse.
"""
import argparse
import copy
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "C:/liars/nn")
import pyliars  # noqa: E402
from common import masked_logits  # noqa: E402
from tnet import TNet, load_tnet, save_tnet, tact, tevaluate  # noqa: E402

OPPS = ("champ", "human", "honestbig", "heavy", "caller", "bluffer", "aggro", "timid", "honest", "random")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="C:/liars/nn/basic_tnet")
    ap.add_argument("--out", default="C:/liars/nn/basic_tnet_ppo")
    ap.add_argument("--champion", default="C:/liars/rust/policy.bin")
    ap.add_argument("--devil", action="store_true")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--kl", type=float, default=0.15)
    ap.add_argument("--ent", type=float, default=0.002)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--habit-boost", type=float, default=20.0,
                    help="step-size multiplier for the player-habit inputs (features 104+), new and still near zero")
    args = ap.parse_args()
    global OPPS
    if args.devil:
        # Fast dumpers, also short-handed (3 left, 1v1), which the advisor defends worst.
        OPPS = OPPS + ("dumper", "dumpliar", "dumper3", "dumper1v1", "ranked", "ranked3", "ranked1v1", "rankedfit", "rankedfit3", "rankedfit1v1")
    OPPS = OPPS + ("reader",)
    dev = "cuda"
    torch.manual_seed(0)
    net = TNet().to(dev)
    load_tnet(net, args.init + ".pt")
    anchor = copy.deepcopy(net).eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    league = [copy.deepcopy(anchor)]
    rng = np.random.default_rng(0)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    env = pyliars.VecEnv(args.envs, args.devil, args.champion, 3)
    ids, f, k, m, sn, h, hl = env.observe()

    def log(msg):
        print(msg, flush=True)

    def run_eval(tag):
        net.eval()
        res = {o: tevaluate(net, o, games=5000, devil=args.devil, champion=args.champion)[0] for o in OPPS}
        net.train()
        log(f"[{tag}] " + " | ".join(f"{o} {res[o]:.3f}" for o in OPPS))
        return res

    start = run_eval("start (copy)")
    best_score = sum(start.values()) / len(start)
    save_tnet(net, args.out)

    for it in range(1, args.iters + 1):
        t0 = time.time()
        net.eval()
        opp_net = league[rng.integers(len(league))]
        steps = []  # per step: (ids, f, k, m, h, hl, actions, logp, value, learner mask)
        for _ in range(args.steps):
            a, logp, v = tact(net, f, k, m, h, hl)
            a_np = a.cpu().numpy()
            snap_rows = np.nonzero(sn)[0]
            if len(snap_rows):
                a2, _, _ = tact(opp_net, f[snap_rows], k[snap_rows], m[snap_rows], h[snap_rows], hl[snap_rows])
                a_np[snap_rows] = a2.cpu().numpy()
            steps.append((ids, f, k, m, h.astype(np.float16), hl, a_np, logp.cpu().numpy(), v.cpu().numpy(), sn == 0))
            ids, f, k, m, sn, h, hl = env.step(a_np.astype(np.uint32).tolist())
        net.train()
        rid, rew = env.drain_rewards()
        # Match every learner decision to its round reward in one vectorized pass.
        ids_all = np.concatenate([st[0] for st in steps]).astype(np.int64)
        learn_all = np.concatenate([st[9] for st in steps])
        order = np.argsort(rid)
        rid_s, rew_s = rid[order].astype(np.int64), rew[order]
        pos = np.searchsorted(rid_s, ids_all)
        pos_c = np.minimum(pos, max(len(rid_s) - 1, 0))
        ok = learn_all & (pos < len(rid_s)) & (rid_s[pos_c] == ids_all) if len(rid_s) else np.zeros_like(learn_all)
        sel = np.nonzero(ok)[0]
        n = len(sel)
        if n == 0:
            continue
        cat = lambda j: np.concatenate([st[j] for st in steps])
        F = torch.from_numpy(cat(1)[sel]).to(dev)
        K = torch.from_numpy(cat(2)[sel].astype(np.int64)).to(dev)
        M = torch.from_numpy(cat(3)[sel]).to(dev)
        H = torch.from_numpy(cat(4)[sel]).to(dev).float()
        HL = torch.from_numpy(cat(5)[sel].astype(np.int64)).to(dev)
        A = torch.from_numpy(cat(6)[sel].astype(np.int64)).to(dev)
        LP = torch.from_numpy(cat(7)[sel]).to(dev)
        V = torch.from_numpy(cat(8)[sel]).to(dev)
        R = torch.from_numpy(rew_s[pos_c[sel]].astype(np.float32)).to(dev)
        adv = R - V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        mb, stats = 8192, []
        for _ in range(3):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, mb):
                i = perm[s:s + mb]
                logits, v = net(F[i], H[i], HL[i])
                logp_all = torch.log_softmax(masked_logits(logits, K[i], M[i]), 1)
                logp = logp_all.gather(1, A[i][:, None]).squeeze(1)
                ratio = torch.exp(logp - LP[i])
                pg = -torch.min(ratio * adv[i], ratio.clamp(0.8, 1.2) * adv[i]).mean()
                vl = ((v - R[i]) ** 2).mean()
                p_all = logp_all.exp()
                ent = -(p_all * logp_all.clamp(min=-30)).sum(1).mean()
                with torch.no_grad():
                    al, _ = anchor(F[i], H[i], HL[i])
                    a_logp = torch.log_softmax(masked_logits(al, K[i], M[i]), 1)
                kl = (p_all * (logp_all.clamp(min=-30) - a_logp.clamp(min=-30))).sum(1).mean()
                loss = pg + 0.5 * vl - args.ent * ent + args.kl * kl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                w0 = net.state[0].weight
                old = w0.data[:, 104:].clone()
                opt.step()
                # Adam ignores gradient scale, so the habit columns get a bigger step directly.
                with torch.no_grad():
                    w0.data[:, 104:] = old + args.habit_boost * (w0.data[:, 104:] - old)
                stats.append((vl.item(), kl.item(), ent.item()))
        s = np.mean(stats, 0)
        env.take_results()
        log(f"iter {it}: {n:,} decisions, value loss {s[0]:.4f}, KL to start {s[1]:.4f}, entropy {s[2]:.3f}, "
            f"mean reward {R.mean().item():+.4f} ({time.time() - t0:.0f}s)")
        if it % 10 == 0:
            league.append(copy.deepcopy(net).eval())
        if it % args.eval_every == 0 or it == args.iters:
            res = run_eval(f"iter {it}")
            worst = min(res[o] - start[o] for o in OPPS)
            score = sum(res.values()) / len(res)
            if worst >= -0.02 and score > best_score:
                best_score = score
                save_tnet(net, args.out)
                log(f"  new best saved to {args.out} (average {score:.3f}, worst change {worst:+.3f})")
            elif worst < -0.02:
                log(f"  not saved: worse against {min(OPPS, key=lambda o: res[o] - start[o])} by {-worst:.3f}")


if __name__ == "__main__":
    main()
