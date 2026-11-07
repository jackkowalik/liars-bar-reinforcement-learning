"""Transformer network: the current situation (104 numbers) plus the last 48 public events.

The situation becomes a summary token placed in front of the event tokens; two small
transformer layers let it look back over the history (who called whom, who lied when, who
dumped real cards...) and the result feeds the same policy heads as the plain network.
Post-norm layers with ReLU, so the Rust side can reproduce them exactly.
"""
import json

import numpy as np
import torch
import torch.nn as nn

from common import N_FEAT, masked_logits

HIST_LEN = 48
EV_DIM = 26  # 20 event features + tells on claims: card checks, think pace, check habit, habit evidence
D = 128


class TNet(nn.Module):
    def __init__(self, d=D, heads=4, layers=2, ff=256):
        super().__init__()
        self.state = nn.Sequential(nn.Linear(N_FEAT, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.cls = nn.Linear(256, d)
        self.ev = nn.Linear(EV_DIM, d)
        self.pos = nn.Parameter(torch.zeros(HIST_LEN + 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, activation="relu", batch_first=True, norm_first=False)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(256 + d, 256), nn.ReLU())
        self.out = nn.Linear(256, 14)
        self.value = nn.Linear(256, 1)

    def forward(self, x, hist, hlen):
        s = self.state(x)
        tok = torch.cat([self.cls(s)[:, None], self.ev(hist)], 1) + self.pos
        # Left-padded history: the first HIST_LEN - hlen event slots are padding.
        idx = torch.arange(HIST_LEN, device=x.device)[None]
        pad = idx < (HIST_LEN - hlen[:, None])
        mask = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        h = self.enc(tok, src_key_padding_mask=mask)[:, 0]
        z = self.head(torch.cat([s, h], 1))
        return self.out(z), self.value(z).squeeze(-1)


@torch.no_grad()
def tact(net, f, k, m, hist, hlen, dev="cuda", greedy=False):
    x = torch.from_numpy(f).to(dev)
    kk = torch.from_numpy(k.astype(np.int64)).to(dev)
    mm = torch.from_numpy(m).to(dev)
    hh = torch.from_numpy(hist).to(dev)
    hl = torch.from_numpy(hlen.astype(np.int64)).to(dev)
    logits, v = net(x, hh, hl)
    lg = masked_logits(logits.float(), kk, mm)
    a = lg.argmax(1) if greedy else torch.distributions.Categorical(logits=lg).sample()
    logp = torch.log_softmax(lg, 1).gather(1, a[:, None]).squeeze(1)
    return a, logp, v.float()


def tevaluate(net, opp, games=4000, n_envs=2048, devil=False, champion="C:/liars/rust/policy.bin", seed=7):
    """Win rate of one transformer seat against three `opp` bots."""
    import pyliars
    env = pyliars.VecEnv(n_envs, devil, champion, seed, eval_opp=opp)
    ids, f, kd, m, sn, h, hl = env.observe()
    done, wins = 0, 0.0
    while done < games:
        a, _, _ = tact(net, f, kd, m, h, hl)
        ids, f, kd, m, sn, h, hl = env.step(a.cpu().numpy().astype(np.uint32).tolist())
        g, w = env.take_results()
        done += g
        wins += w
        env.drain_rewards()
    return wins / done, done


def load_tnet(net, path):
    """Loads a saved state dict; networks from before the tells get zero weights for them, so
    they start out playing exactly as before."""
    sd = torch.load(path)
    for key, dim in (("ev.weight", EV_DIM), ("state.0.weight", N_FEAT)):
        w = sd[key]
        if w.shape[1] < dim:
            sd[key] = torch.cat([w, torch.zeros(w.shape[0], dim - w.shape[1], dtype=w.dtype, device=w.device)], 1)
    net.load_state_dict(sd)
    return net


def save_tnet(net, base):
    """Raw little-endian f32 export, every parameter in named order, plus the shapes."""
    names, shapes = [], []
    with open(base + ".bin", "wb") as f:
        for n, p in net.state_dict().items():
            a = p.detach().float().cpu().numpy().astype("<f4")
            f.write(a.tobytes())
            names.append(n)
            shapes.append(list(a.shape))
    json.dump({"kind": "tnet", "d": D, "hist_len": HIST_LEN, "ev_dim": EV_DIM, "params": list(zip(names, shapes))},
              open(base + ".json", "w"))
    torch.save(net.state_dict(), base + ".pt")
