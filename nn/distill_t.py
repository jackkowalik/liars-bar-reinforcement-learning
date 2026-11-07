"""Warm start for the transformer: imitate the current best plain network.

The teacher (live MLP) plays games in the batched environment; the transformer learns to match
its move probabilities in every situation it meets, with the event history available. It should
end up about as strong as the teacher before self-play improves it further.
"""
import argparse
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "C:/liars/nn")
import pyliars  # noqa: E402
from common import Net, act, load_raw, masked_logits  # noqa: E402
from tnet import TNet, save_tnet, tevaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="C:/liars/nn/basic_live")
    ap.add_argument("--champion", default="C:/liars/rust/policy.bin")
    ap.add_argument("--devil", action="store_true")
    ap.add_argument("--out", default="C:/liars/nn/basic_tnet")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--envs", type=int, default=4096)
    args = ap.parse_args()
    dev = "cuda"
    teacher = load_raw(Net(), args.teacher).to(dev).eval()
    net = TNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=1e-5)
    env = pyliars.VecEnv(args.envs, args.devil, args.champion, 11)
    ids, f, k, m, sn, h, hl = env.observe()
    print(f"transformer: {sum(p.numel() for p in net.parameters()):,} weights", flush=True)
    for it in range(1, args.iters + 1):
        t0 = time.time()
        F, K, M, H, HL = [], [], [], [], []
        for _ in range(args.steps):
            F.append(f); K.append(k); M.append(m); H.append(h); HL.append(hl)
            a, _, _ = act(teacher, f, k, m)
            ids, f, k, m, sn, h, hl = env.step(a.cpu().numpy().astype(np.uint32).tolist())
            env.drain_rewards()
        X = torch.from_numpy(np.concatenate(F)).to(dev)
        Kt = torch.from_numpy(np.concatenate(K).astype(np.int64)).to(dev)
        Mt = torch.from_numpy(np.concatenate(M)).to(dev)
        Ht = torch.from_numpy(np.concatenate(H)).to(dev)
        HLt = torch.from_numpy(np.concatenate(HL).astype(np.int64)).to(dev)
        with torch.no_grad():
            tl, _ = teacher(X)
            tp = torch.softmax(masked_logits(tl, Kt, Mt), 1)
        n, mb, losses = len(X), 8192, []
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, mb):
            i = perm[s:s + mb]
            logits, _ = net(X[i], Ht[i], HLt[i])
            logp = torch.log_softmax(masked_logits(logits, Kt[i], Mt[i]), 1)
            loss = -(tp[i] * logp.clamp(min=-30)).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        ent = -(tp * torch.log(tp.clamp(min=1e-9))).sum(1).mean().item()
        print(f"iter {it}: {n:,} states, cross-entropy {np.mean(losses):.4f} (teacher entropy {ent:.4f}, gap = KL) ({time.time() - t0:.0f}s)", flush=True)
        if it % 10 == 0 or it == args.iters:
            net.eval()
            res = {o: tevaluate(net, o, games=4000, devil=args.devil, champion=args.champion)[0] for o in ("champ", "human", "caller", "honest")}
            net.train()
            print("   eval: " + " | ".join(f"{o} {r:.3f}" for o, r in res.items()), flush=True)
            save_tnet(net, args.out)


if __name__ == "__main__":
    main()
