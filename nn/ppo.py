"""Stage 3: improve the network by self-play (PPO), starting from the champion imitation.

Each network decision is paid the change in that seat's win probability over its round (the
same reward the table trainer uses). Opponents: log-fitted human-like players, champion table
copies, a probability bot, and the network itself. A KL leash to the imitation network keeps it
from drifting into strange play while it improves.
"""
import argparse
import copy
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "C:/liars/nn")
import pyliars  # noqa: E402
from common import Net, act, evaluate, load_raw, masked_logits, save_raw  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default="C:/liars/nn/basic_net")
    ap.add_argument("--out", default="C:/liars/nn/basic_ppo")
    ap.add_argument("--champion", default="C:/liars/rust/policy.bin")
    ap.add_argument("--devil", action="store_true")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=256, help="env steps per iteration")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--kl", type=float, default=0.15, help="leash strength to the imitation policy")
    ap.add_argument("--ent", type=float, default=0.002)
    ap.add_argument("--eval-every", type=int, default=5)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)

    net = load_raw(Net(), args.init).to(dev)
    anchor = copy.deepcopy(net).eval()
    for p in anchor.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    env = pyliars.VecEnv(args.envs, args.devil, args.champion, 1)
    ids, f, k, m, sn, *_ = env.observe()
    # League: frozen past versions play some seats (starting with the champion copy).
    league = [copy.deepcopy(anchor)]
    rng = np.random.default_rng(0)

    # Decisions waiting for their round's reward: id -> (features, kind, mask, action, logp, value).
    pending = {}

    def log(msg):
        print(msg, flush=True)

    opps = ("champ", "human", "honestbig", "heavy", "caller", "bluffer", "aggro", "timid", "honest", "random")

    def run_eval(tag):
        net.eval()
        res = {o: evaluate(net, o, games=5000, devil=args.devil, champion=args.champion)[0] for o in opps}
        net.train()
        log(f"[{tag}] " + " | ".join(f"{o} {res[o]:.3f}" for o in opps))
        return res

    start = run_eval("start (imitation)")
    best_score = sum(start.values()) / len(start)
    save_raw(net, args.out)

    for it in range(1, args.iters + 1):
        t0 = time.time()
        net.eval()
        opp_net = league[rng.integers(len(league))]
        for _ in range(args.steps):
            a, logp, v = act(net, f, k, m)
            a_np = a.cpu().numpy()
            snap_rows = np.nonzero(sn)[0]
            if len(snap_rows):
                a2, _, _ = act(opp_net, f[snap_rows], k[snap_rows], m[snap_rows])
                a_np[snap_rows] = a2.cpu().numpy()
            lp, vv = logp.cpu().numpy(), v.cpu().numpy()
            for j, i in enumerate(ids):
                if not sn[j]:
                    pending[int(i)] = (f[j], k[j], m[j], int(a_np[j]), float(lp[j]), float(vv[j]))
            ids, f, k, m, sn, *_ = env.step(a_np.astype(np.uint32).tolist())
        net.train()
        rid, rew = env.drain_rewards()
        batch = [pending.pop(int(i)) + (float(r),) for i, r in zip(rid, rew) if int(i) in pending]
        if len(pending) > 5_000_000:
            pending.clear()
        if not batch:
            continue
        F = torch.from_numpy(np.stack([b[0] for b in batch])).to(dev)
        K = torch.tensor([b[1] for b in batch], device=dev, dtype=torch.long)
        M = torch.from_numpy(np.stack([b[2] for b in batch])).to(dev)
        A = torch.tensor([b[3] for b in batch], device=dev)
        LP = torch.tensor([b[4] for b in batch], device=dev)
        V = torch.tensor([b[5] for b in batch], device=dev)
        R = torch.tensor([b[6] for b in batch], device=dev)
        adv = R - V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = len(batch)
        mb = 16384
        stats = []
        for _ in range(3):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, mb):
                idx = perm[s:s + mb]
                logits, v = net(F[idx])
                lg = masked_logits(logits, K[idx], M[idx])
                logp_all = torch.log_softmax(lg, 1)
                logp = logp_all.gather(1, A[idx][:, None]).squeeze(1)
                ratio = torch.exp(logp - LP[idx])
                pg = -torch.min(ratio * adv[idx], ratio.clamp(0.8, 1.2) * adv[idx]).mean()
                vl = ((v - R[idx]) ** 2).mean()
                p_all = logp_all.exp()
                ent = -(p_all * logp_all.clamp(min=-30)).sum(1).mean()
                with torch.no_grad():
                    a_logits, _ = anchor(F[idx])
                    a_logp = torch.log_softmax(masked_logits(a_logits, K[idx], M[idx]), 1)
                kl = (p_all * (logp_all.clamp(min=-30) - a_logp.clamp(min=-30))).sum(1).mean()
                loss = pg + 0.5 * vl - args.ent * ent + args.kl * kl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                stats.append((pg.item(), vl.item(), kl.item(), ent.item()))
        s = np.mean(stats, 0)
        g, w = env.take_results()
        log(f"iter {it}: {n:,} decisions, value loss {s[1]:.4f}, KL to champion-copy {s[2]:.4f}, "
            f"entropy {s[3]:.3f}, mean reward {R.mean().item():+.4f} ({time.time() - t0:.0f}s)")
        if it % 10 == 0:
            league.append(copy.deepcopy(net).eval())
        if it % args.eval_every == 0 or it == args.iters:
            res = run_eval(f"iter {it}")
            save_raw(net, args.out + f"_it{it}")
            # Gate: no opponent type may be more than 2 points worse than the champion copy.
            worst = min(res[o] - start[o] for o in opps)
            score = sum(res.values()) / len(res)
            if worst >= -0.02 and score > best_score:
                best_score = score
                save_raw(net, args.out)
                log(f"  new best saved to {args.out} (average {score:.3f}, worst change {worst:+.3f})")
            elif worst < -0.02:
                log(f"  not saved: worse against {min(opps, key=lambda o: res[o] - start[o])} by {-worst:.3f}")


if __name__ == "__main__":
    main()
