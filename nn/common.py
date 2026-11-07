"""Shared network definition and loading/saving for the Liar's Deck networks."""
import json

import numpy as np
import torch
import torch.nn as nn

N_FEAT = 135
HEAD = {0: (0, 2), 1: (2, 4), 2: (4, 14)}


class Net(nn.Module):
    """Shared trunk, 14 policy logits (deal 2, call 2, play 10) and a value head."""

    def __init__(self, hidden=(512, 512, 256)):
        super().__init__()
        layers, d = [], N_FEAT
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(d, 14)
        self.value = nn.Linear(d, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.out(h), self.value(h).squeeze(-1)


def masked_logits(logits, kind, mask):
    """Each row's head padded to 10 slots, illegal slots at -1e9."""
    out = torch.full((len(kind), 10), -1e9, device=logits.device, dtype=logits.dtype)
    for k, (a, b) in HEAD.items():
        sel = kind == k
        if sel.any():
            out[sel, : b - a] = logits[sel, a:b]
    return out.masked_fill(mask < 0.5, -1e9)


def load_raw(net, base):
    """Loads the policy part from a raw export (<base>.bin/.json) written by train_imitate."""
    meta = json.load(open(base + ".json"))
    raw = np.fromfile(base + ".bin", dtype="<f4")
    linears = [m for m in net.trunk if isinstance(m, nn.Linear)] + [net.out]
    at = 0
    for m, (i, o) in zip(linears, meta["layers"]):
        m.weight.data = torch.from_numpy(raw[at:at + i * o].reshape(o, i).copy()); at += i * o
        m.bias.data = torch.from_numpy(raw[at:at + o].copy()); at += o
    assert at == raw.size
    return net


def save_raw(net, base):
    """Exports the policy part in the raw format the Rust side loads."""
    linears = [m for m in net.trunk if isinstance(m, nn.Linear)] + [net.out]
    with open(base + ".bin", "wb") as f:
        for m in linears:
            f.write(m.weight.detach().float().cpu().numpy().astype("<f4").tobytes())
            f.write(m.bias.detach().float().cpu().numpy().astype("<f4").tobytes())
    json.dump({"n_feat": N_FEAT, "layers": [[m.in_features, m.out_features] for m in linears],
               "heads": {"deal": [0, 2], "call": [2, 4], "play": [4, 14]}}, open(base + ".json", "w"))


@torch.no_grad()
def act(net, feats, kinds, masks, dev="cuda", greedy=False):
    x = torch.from_numpy(feats).to(dev)
    k = torch.from_numpy(kinds.astype(np.int64)).to(dev)
    m = torch.from_numpy(masks).to(dev)
    logits, v = net(x)
    lg = masked_logits(logits.float(), k, m)
    if greedy:
        a = lg.argmax(1)
    else:
        a = torch.distributions.Categorical(logits=lg).sample()
    logp = torch.log_softmax(lg, 1).gather(1, a[:, None]).squeeze(1)
    return a, logp, v.float()


def evaluate(net, opp, games=4000, n_envs=2048, devil=False, champion="C:/liars/rust/policy.bin", seed=7):
    """Win rate of one network seat against three `opp` bots."""
    import pyliars
    env = pyliars.VecEnv(n_envs, devil, champion, seed, eval_opp=opp)
    ids, f, kd, m, *_ = env.observe()
    done, wins = 0, 0.0
    while done < games:
        a, _, _ = act(net, f, kd, m)
        ids, f, kd, m, *_ = env.step(a.cpu().numpy().astype(np.uint32).tolist())
        g, w = env.take_results()
        done += g
        wins += w
        env.drain_rewards()
    return wins / done, done
