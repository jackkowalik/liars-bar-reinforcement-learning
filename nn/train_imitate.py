"""Train a small network to imitate the champion table policy (stage 2 of the NN plan).

Data: rows written by `liars gen-data`: kind, 104 features, 10 target probs, 10 legal mask.
Output: weights as raw little-endian f32 for the Rust forward pass, plus a JSON description.
"""
import argparse
import glob
import json
import time

import numpy as np
import torch
import torch.nn as nn

N_FEAT = 104
ROW = 1 + N_FEAT + 20
# Output slots: deal (2), call (2), play (10).
HEAD = {0: (0, 2), 1: (2, 4), 2: (4, 14)}


class Net(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        layers, d = [], N_FEAT
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(d, 14)

    def forward(self, x):
        return self.out(self.trunk(x))


def masked_logits(logits, kind, mask):
    """Picks each row's head and pads to 10 slots, with illegal slots at -inf."""
    out = torch.full((len(kind), 10), -1e9, device=logits.device)
    for k, (a, b) in HEAD.items():
        sel = kind == k
        if sel.any():
            out[sel, : b - a] = logits[sel, a:b]
    return out.masked_fill(mask < 0.5, -1e9)


def load(pattern):
    parts = [np.fromfile(f, dtype="<f4").reshape(-1, ROW) for f in sorted(glob.glob(pattern))]
    return np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="C:/liars/nn/basic.bin.*")
    ap.add_argument("--out", default="C:/liars/nn/basic_net")
    ap.add_argument("--hidden", default="512,512,256")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    dev = "cuda"
    t0 = time.time()
    data = load(args.data)
    np.random.default_rng(0).shuffle(data)
    n_val = max(10000, len(data) // 100)
    val, train = data[:n_val], data[n_val:]
    print(f"{len(data):,} samples loaded in {time.time() - t0:.0f}s")

    def to_dev(a):
        return (torch.from_numpy(a[:, 0]).to(dev).long(),
                torch.from_numpy(a[:, 1:1 + N_FEAT]).to(dev).half(),
                torch.from_numpy(a[:, 1 + N_FEAT:1 + N_FEAT + 10]).to(dev).half(),
                torch.from_numpy(a[:, 1 + N_FEAT + 10:]).to(dev).half())

    tk, tx, ty, tm = to_dev(train)
    vk, vx, vy, vm = to_dev(val)
    del data, train, val
    hidden = [int(h) for h in args.hidden.split(",")]
    net = Net(hidden).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    steps = args.epochs * (len(tk) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)

    def loss_on(k, x, y, m):
        logp = torch.log_softmax(masked_logits(net(x.float()), k, m.float()), dim=1)
        y = y.float() * m.float()
        return -(y * logp.clamp(min=-30)).sum(1).mean(), logp

    def evaluate():
        net.eval()
        with torch.no_grad():
            tot, agree, kl = 0.0, 0.0, 0.0
            for i in range(0, len(vk), 65536):
                s = slice(i, i + 65536)
                l, logp = loss_on(vk[s], vx[s], vy[s], vm[s])
                y = vy[s].float() * vm[s].float()
                agree += (logp.argmax(1) == y.argmax(1)).float().sum().item()
                ent = -(y * torch.log(y.clamp(min=1e-9))).sum(1)
                kl += (l * len(y) - ent.sum()).item()
                tot += len(y)
        net.train()
        return agree / tot, kl / tot

    step = 0
    for ep in range(args.epochs):
        perm = torch.randperm(len(tk), device=dev)
        t1 = time.time()
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            idx = perm[i:i + args.batch]
            l, _ = loss_on(tk[idx], tx[idx], ty[idx], tm[idx])
            opt.zero_grad(set_to_none=True)
            l.backward()
            opt.step()
            sched.step()
            step += 1
        agree, kl = evaluate()
        print(f"epoch {ep + 1}/{args.epochs}: top choice matches champion {agree:.1%}, "
              f"avg KL {kl:.4f} ({time.time() - t1:.0f}s)", flush=True)

    # Export: layers in order, each as out x in weights then biases, little-endian f32.
    layers = [m for m in net.modules() if isinstance(m, nn.Linear)]
    with open(args.out + ".bin", "wb") as f:
        for m in layers:
            f.write(m.weight.detach().float().cpu().numpy().astype("<f4").tobytes())
            f.write(m.bias.detach().float().cpu().numpy().astype("<f4").tobytes())
    meta = {"n_feat": N_FEAT, "layers": [[m.in_features, m.out_features] for m in layers],
            "heads": {"deal": [0, 2], "call": [2, 4], "play": [4, 14]}}
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f)
    print(f"saved {args.out}.bin / .json ({sum(p.numel() for p in net.parameters()):,} weights)")


if __name__ == "__main__":
    main()
