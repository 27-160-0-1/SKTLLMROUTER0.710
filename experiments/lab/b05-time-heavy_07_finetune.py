# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b05 step 7: fine-tune a small multilingual encoder on the two gains.

The gate demanded by the brief: measure the encoder's OUT-OF-FOLD gain-axis
quality first.  Same 10 folds and the same train-only rows as the b05base
stage, so the numbers are directly comparable with the hashed-feature heads.

  python b05-time-heavy_07_finetune.py <preset> [...]
presets: gain level multi rank long
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
from transformers import AutoTokenizer, AutoModel     # noqa: E402

MODEL = "intfloat/multilingual-e5-small"
OUTJ = Path("reports/lab/b05_finetune.json")
CACHE = Path("reports/lab/b05_ft_oof.npz")
DEVICE = torch.device("cuda")

lab = lib.XLab(verbose=False)
TEXTS = lab.texts
TG = lab.targets                    # (n,6) 3 scores + 3 log costs
D = lab.delta_targets               # (n,2) d1, d2
TRAIN = lab.train_idx
FOLD = np.random.default_rng(123).integers(0, 10, size=len(TRAIN))
RESULTS = json.loads(OUTJ.read_text()) if OUTJ.exists() else []
STORE = dict(np.load(CACHE)) if CACHE.exists() else {}

tok = AutoTokenizer.from_pretrained(MODEL)


def encode(maxlen):
    ids = tok(TEXTS, add_special_tokens=True, truncation=True, max_length=maxlen)["input_ids"]
    return ids


class Head(nn.Module):
    def __init__(self, nout, hidden=384):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL)
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden, nout)

    def forward(self, ii, am):
        h = self.enc(input_ids=ii, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).float()
        p = (h * m).sum(1) / m.sum(1)
        return self.fc(self.drop(p))


def batches(idx, ids, bs, shuffle, rng=None):
    order = np.array(idx)
    if shuffle:
        order = order[rng.permutation(len(order))]
    else:
        order = order[np.argsort([len(ids[i]) for i in order])]
    for s in range(0, len(order), bs):
        sel = order[s:s + bs]
        L = max(len(ids[i]) for i in sel)
        ii = torch.full((len(sel), L), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(sel), L), dtype=torch.long)
        for j, i in enumerate(sel):
            b = ids[i]; ii[j, :len(b)] = torch.tensor(b); am[j, :len(b)] = 1
        yield sel, ii.to(DEVICE), am.to(DEVICE)


def fit_fold(fit_idx, hold_idx, ids, targets, epochs, lr, bs, seed, ranking=False):
    torch.manual_seed(seed)
    y = targets[fit_idx]
    mu, sd = y.mean(axis=0), y.std(axis=0) + 1e-9
    model = Head(targets.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    nstep = epochs * int(np.ceil(len(fit_idx) / bs))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(nstep, 1),
                                              pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda")
    T = torch.tensor((targets - mu) / sd, dtype=torch.float32, device=DEVICE)
    rng = np.random.default_rng(seed)
    model.train()
    for _ep in range(epochs):
        for sel, ii, am in batches(fit_idx, ids, bs, True, rng):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(ii, am)
                tt = T[sel]
                if ranking:
                    # pairwise logistic (RankNet) inside the batch, per column,
                    # weighted by the true gap
                    loss = 0.0
                    for k in range(out.shape[1]):
                        dp = out[:, k][:, None] - out[:, k][None, :]
                        dy = tt[:, k][:, None] - tt[:, k][None, :]
                        w = dy.abs()
                        loss = loss + (w * torch.nn.functional.softplus(-torch.sign(dy) * dp)
                                       ).sum() / (w.sum() + 1e-6)
                    loss = loss / out.shape[1]
                else:
                    loss = nn.functional.mse_loss(out, tt)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sch.step()
    model.eval()
    pred = np.zeros((len(hold_idx), targets.shape[1]))
    pos = {int(v): k for k, v in enumerate(hold_idx)}
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for sel, ii, am in batches(hold_idx, ids, 32, False):
            o = model(ii, am).float().cpu().numpy() * sd + mu
            for j, i in enumerate(sel):
                pred[pos[int(i)]] = o[j]
    del model, opt
    torch.cuda.empty_cache()
    return pred


def run(name, targets, gcols, epochs=4, lr=2e-5, bs=16, maxlen=256, seeds=(0,), ranking=False):
    """gcols: (i,j) columns of `targets` that carry (d1, d2), or a callable."""
    ids = encode(maxlen)
    t0 = time.perf_counter()
    oof = np.zeros((len(TRAIN), targets.shape[1]))
    for k in range(10):
        fi = TRAIN[FOLD != k]; hi = TRAIN[FOLD == k]
        acc = np.zeros((len(hi), targets.shape[1]))
        for s in seeds:
            acc += fit_fold(fi, hi, ids, targets, epochs, lr, bs, s, ranking)
        oof[FOLD == k] = acc / len(seeds)
    dev = np.zeros((len(lab.dev_idx), targets.shape[1]))
    for s in seeds:
        dev += fit_fold(TRAIN, lab.dev_idx, ids, targets, epochs, lr, bs, s, ranking)
    dev /= len(seeds)
    secs = time.perf_counter() - t0
    g1, g2 = (oof[:, gcols[0]], oof[:, gcols[1]]) if not callable(gcols) else gcols(oof)
    e1, e2 = (dev[:, gcols[0]], dev[:, gcols[1]]) if not callable(gcols) else gcols(dev)
    dg = lib.gain_axis(lab, TRAIN, g1, g2)
    dd = lib.gain_axis(lab, lab.dev_idx, e1, e2)
    STORE[name + "|dev"] = np.column_stack([e1, e2])
    row = dict(name=name, secs=round(secs, 1), epochs=epochs, lr=lr, bs=bs, maxlen=maxlen,
               seeds=len(seeds), **{k: round(v, 4) for k, v in dg.items()},
               **{"dev_" + k: round(v, 4) for k, v in dd.items()})
    RESULTS.append(row)
    OUTJ.write_text(json.dumps(RESULTS, indent=1), encoding="utf-8")
    STORE[name] = np.column_stack([g1, g2])
    np.savez_compressed(CACHE, **STORE)
    print(f"{name:34s}{secs:7.1f}s OOF corr d1={dg['corr1']:+.4f} d2={dg['corr2']:+.4f} "
          f"wfAUC1={dg['auc1']:.4f} wfAUC2={dg['auc2']:.4f} | DEV d1={dd['corr1']:+.4f} "
          f"d2={dd['corr2']:+.4f} A1={dd['auc1']:.4f} A2={dd['auc2']:.4f}", flush=True)
    return oof


PRESETS = {
    "gain":  lambda: run("F1 e5s ft -> (d1,d2) 4ep", D, (0, 1)),
    "level": lambda: run("F2 e5s ft -> 3 levels 4ep", TG[:, :3], lambda o: (o[:, 1] - o[:, 0],
                                                                            o[:, 2] - o[:, 1])),
    "multi": lambda: run("F3 e5s ft -> 3 lvl + 2 gain", np.column_stack([TG[:, :3], D]), (3, 4)),
    "rank":  lambda: run("F4 e5s ft pairwise rank", D, (0, 1), ranking=True),
    "long":  lambda: run("F5 e5s ft (d1,d2) 12ep 3sd", D, (0, 1), epochs=12, seeds=(0, 1, 2)),
    "len512": lambda: run("F6 e5s ft (d1,d2) maxlen512", D, (0, 1), maxlen=512, bs=8),
}

if __name__ == "__main__":
    print(f"reference (hashed 58-feature heads, same rows/folds):\n"
          f"  deployed GBM gain : corr d1=+0.0344 d2=+0.4262 wfAUC1=0.4874 wfAUC2=0.5416\n"
          f"  ridge a=30        : corr d1=+0.0761 d2=+0.4353 wfAUC1=0.5182 wfAUC2=0.5442",
          flush=True)
    for p in (sys.argv[1:] or ["gain"]):
        PRESETS[p]()
