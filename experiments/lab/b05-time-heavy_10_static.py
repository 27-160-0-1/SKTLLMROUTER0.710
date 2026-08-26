# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b05 step 10: the representation that ACTUALLY fits the container.

Step 8 measured the arithmetic: a token-embedding lookup + mean pool over the
2,640 episodes is 0.4 GFLOP (0.34 s at 2 threads on the dev box) against a
budget of 90 s x 2 cores, while a 12-layer d=384 encoder is 9.4-15.9 TFLOP
(141-253 s at 2 threads).  So the static model is the only encoder-family
representation with a real deployment path.

Here the static model is TRAINED (on GPU, cheap) directly on the two gains,
fold-pure over the same 10 folds, and judged on the same gain axis.

  python b05-time-heavy_10_static.py <preset> [...]
"""
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("b05lib", HERE / "b05-time-heavy_lib.py")
lib = importlib.util.module_from_spec(spec); sys.modules["b05lib"] = lib
spec.loader.exec_module(lib)
from transformers import AutoTokenizer, AutoModel  # noqa: E402

MODEL = "intfloat/multilingual-e5-small"
DEVICE = torch.device("cuda")
OUTJ = Path("reports/lab/b05_static.json")
CACHE = Path("reports/lab/b05_static_oof.npz")

lab = lib.XLab(verbose=False)
TRAIN = lab.train_idx
FOLD = np.random.default_rng(123).integers(0, 10, size=len(TRAIN))
D = lab.delta_targets
TG = lab.targets
ROWS = json.loads(OUTJ.read_text()) if OUTJ.exists() else []
STORE = dict(np.load(CACHE)) if CACHE.exists() else {}

tok = AutoTokenizer.from_pretrained(MODEL)
CAP = 512
IDS = tok(lab.texts, add_special_tokens=True, truncation=True, max_length=CAP)["input_ids"]
NLEN = np.array([len(s) for s in IDS])
PAD = np.zeros((len(IDS), int(NLEN.max())), dtype=np.int64)
for i, s in enumerate(IDS):
    PAD[i, :len(s)] = s
PAD_T = torch.from_numpy(PAD)
LEN_T = torch.from_numpy(NLEN)
E0 = AutoModel.from_pretrained(MODEL).embeddings.word_embeddings.weight.detach().clone()
V, DIM = E0.shape
print(f"vocab={V} dim={DIM}  total tokens (cap {CAP}) = {int(NLEN.sum())}", flush=True)


def bag(sel):
    """(ids, mask) for a length-bucketed batch of episode indices."""
    sel = np.asarray(sel)
    L = int(NLEN[sel].max())
    ids = PAD_T[sel, :L].to(DEVICE, non_blocking=True)
    ln = LEN_T[sel].to(DEVICE, non_blocking=True)
    mask = (torch.arange(L, device=DEVICE)[None, :] < ln[:, None]).float()
    return ids, mask


class Static(nn.Module):
    """Token-embedding lookup + mean pool + head.  Everything above the lookup
    can be folded into the table offline, so inference is one gather per token."""

    def __init__(self, nout, rank=None, train_table=False, hidden=0):
        super().__init__()
        self.emb = nn.Embedding(V, DIM, _weight=E0.clone())
        self.emb.weight.requires_grad_(train_table)
        layers = []
        d = DIM
        if rank:
            layers += [nn.Linear(DIM, rank)]; d = rank
        if hidden:
            layers += [nn.GELU(), nn.Linear(d, hidden), nn.GELU()]; d = hidden
        layers += [nn.Linear(d, nout)]
        self.head = nn.Sequential(*layers)

    def forward(self, ids, mask):
        e = self.emb(ids) * mask.unsqueeze(-1)
        return self.head(e.sum(1) / mask.sum(1, keepdim=True))


def fit_fold(fit_idx, hold_idx, targets, epochs, lr, bs, seed, **mk):
    torch.manual_seed(seed)
    y = targets[fit_idx]
    mu, sd = y.mean(axis=0), y.std(axis=0) + 1e-9
    T = torch.tensor((targets - mu) / sd, dtype=torch.float32, device=DEVICE)
    m = Static(targets.shape[1], **mk).to(DEVICE)
    ps = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.01)
    nstep = epochs * int(np.ceil(len(fit_idx) / bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(nstep, 1),
                                              pct_start=0.2)
    rng = np.random.default_rng(seed)
    m.train()
    for _ in range(epochs):
        order = np.array(fit_idx)[rng.permutation(len(fit_idx))]
        for s in range(0, len(order), bs):
            sel = order[s:s + bs]
            f, o = bag(sel)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(m(f, o), T[sel])
            loss.backward(); opt.step(); sch.step()
    m.eval()
    out = np.zeros((len(hold_idx), targets.shape[1]))
    with torch.no_grad():
        for s in range(0, len(hold_idx), 64):
            sel = np.array(hold_idx)[s:s + 64]
            f, o = bag(sel)
            out[s:s + 64] = m(f, o).cpu().numpy() * sd + mu
    return out


def run(name, targets, gcols, epochs=30, lr=3e-3, bs=64, seeds=(0, 1, 2), **mk):
    t0 = time.perf_counter()
    oof = np.zeros((len(TRAIN), targets.shape[1]))
    for k in range(10):
        fi = TRAIN[FOLD != k]; hi = TRAIN[FOLD == k]
        acc = sum(fit_fold(fi, hi, targets, epochs, lr, bs, s, **mk) for s in seeds) / len(seeds)
        oof[FOLD == k] = acc
    dev = sum(fit_fold(TRAIN, lab.dev_idx, targets, epochs, lr, bs, s, **mk)
              for s in seeds) / len(seeds)
    secs = time.perf_counter() - t0
    g = (lambda o: (o[:, gcols[0]], o[:, gcols[1]])) if not callable(gcols) else gcols
    g1, g2 = g(oof); e1, e2 = g(dev)
    dg = lib.gain_axis(lab, TRAIN, g1, g2)
    dd = lib.gain_axis(lab, lab.dev_idx, e1, e2)
    row = dict(name=name, secs=round(secs, 1), epochs=epochs, lr=lr, seeds=len(seeds), **mk,
               **{k: round(v, 4) for k, v in dg.items()},
               **{"dev_" + k: round(v, 4) for k, v in dd.items()})
    ROWS.append(row); OUTJ.write_text(json.dumps(ROWS, indent=1), encoding="utf-8")
    STORE[name] = np.column_stack([g1, g2]); STORE[name + "|dev"] = np.column_stack([e1, e2])
    np.savez_compressed(CACHE, **STORE)
    print(f"{name:34s}{secs:7.1f}s OOF d1={dg['corr1']:+.4f} d2={dg['corr2']:+.4f} "
          f"A1={dg['auc1']:.4f} A2={dg['auc2']:.4f} | DEV d1={dd['corr1']:+.4f} "
          f"d2={dd['corr2']:+.4f} A1={dd['auc1']:.4f} A2={dd['auc2']:.4f}", flush=True)


PRESETS = {
    "frozen":  lambda: run("S1 frozen table + linear", D, (0, 1)),
    "rank64":  lambda: run("S2 frozen + rank64 + MLP", D, (0, 1), rank=64, hidden=64),
    "trained": lambda: run("S3 trainable table + linear", D, (0, 1), train_table=True, lr=1e-3),
    "multi":   lambda: run("S4 frozen, 3 lvl + 2 gain", np.column_stack([TG[:, :3], D]), (3, 4)),
    "multitr": lambda: run("S5 trainable, 3lvl+2gain", np.column_stack([TG[:, :3], D]), (3, 4),
                           train_table=True, lr=1e-3),
}

if __name__ == "__main__":
    for p in (sys.argv[1:] or ["frozen"]):
        PRESETS[p]()
