# SPDX-License-Identifier: Apache-2.0
"""a08 probe 3: (a) honest counterfactual for an exact input-token feature,
(b) how well cheap statistics / a real tokenizer predict input tokens,
(c) family-constant instruction offsets (E41 claim) reproduced from our own text.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from labdata import (load_all, MODEL_IDS, RATES, TOKEN_UNIT, TIERS, TIER_WEIGHT,
                     tier_result)  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

tr, dv = load_all()
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
IN_RATE = np.array([RATES[m][1] for m in MODEL_IDS]) / TOKEN_UNIT
OUT_RATE = np.array([RATES[m][2] for m in MODEL_IDS]) / TOKEN_UNIT
sp = dv
cin_true = sp.itok * IN_RATE[None, :]
cout_true = sp.otok * OUT_RATE[None, :]


def best_safety(mkc, label):
    tot = 0.0
    parts = []
    for t in TIERS:
        best = None
        for s in np.arange(0.60, 1.401, 0.005):
            r = tier_result(P[f"score_{t}"], mkc(t), sp, t, float(s))
            if r["passed"] and (best is None or r["score"] > best[0]):
                best = (r["score"], float(s))
        tot += TIER_WEIGHT[t] * best[0]
        parts.append(f"{t[:4]}={best[0]:.4f}@{best[1]:.3f}")
    print(f"  {label:56s} {tot:.4f}  " + " ".join(parts))
    return tot


print("=== (a) exact-input-token counterfactuals (dev, best safety per tier) ===")
best_safety(lambda t: P[f"cost_{t}"], "deployed pred cost")


def cf_keep_relerr(t):
    """current multiplicative error on the TOTAL, applied to the OUTPUT part only."""
    ratio = P[f"cost_{t}"] / sp.cost
    return cin_true + cout_true * ratio


def cf_subtract(t):
    return cin_true + np.maximum(P[f"cost_{t}"] - cin_true, 1e-9)


best_safety(cf_keep_relerr, "exact cin, current rel-err on cout only")
best_safety(cf_subtract, "exact cin, implied cout = pred_total - cin")
best_safety(lambda t: sp.cost, "true cost")

# how often does pred_total fall below the exact input part?
for t in TIERS:
    bad = (P[f"cost_{t}"] < cin_true).mean(axis=0)
    print(f"    {t:9s} frac(pred_total < exact input part) per model = {np.round(bad,3)}")

print()
print("=== (b) input-token predictability from cheap text statistics ===")
import re  # noqa: E402
_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)


def stats(text):
    n = len(text)
    ws = _SPACE.sub(" ", text)
    words = _WORD.findall(text)
    nb = len(text.encode("utf-8"))
    hang = sum(1 for c in text if "가" <= c <= "힣")
    dig = sum(c.isdigit() for c in text)
    nonascii = sum(1 for c in text if ord(c) >= 128)
    punct = sum(1 for c in text if (not c.isalnum()) and (not c.isspace()))
    return [n, nb, len(ws), len(words), text.count("\n"), hang, dig, nonascii, punct,
            sum(len(w) for w in words), n - len(ws)]


def fit_eval(texts_tr, y_tr, fam_tr, texts_te, y_te, fam_te, fams):
    Xtr = np.array([stats(t) for t in texts_tr], dtype=float)
    Xte = np.array([stats(t) for t in texts_te], dtype=float)
    Ftr = np.array([[float(f == g) for g in fams] for f in fam_tr])
    Fte = np.array([[float(f == g) for g in fams] for f in fam_te])
    A = np.hstack([Xtr, Ftr, np.ones((len(Xtr), 1))])
    B = np.hstack([Xte, Fte, np.ones((len(Xte), 1))])
    coef, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    pred = B @ coef
    ss = 1 - ((pred - y_te) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum()
    ape = np.abs(pred - y_te) / np.maximum(y_te, 1)
    return ss, np.median(ape), pred.sum() / y_te.sum(), pred


fam_tr = [classify_family(t) for t in tr.texts]
fam_dv = [classify_family(t) for t in dv.texts]
fams = sorted(set(fam_tr))
y_tr = tr.itok[:, 0]
y_dv = dv.itok[:, 0]
r2, mape, sr, pred_it = fit_eval(tr.texts, y_tr, fam_tr, dv.texts, y_dv, fam_dv, fams)
print(f"  11 cheap stats + family one-hot -> itok(light):  dev R2={r2:.4f} "
      f"medAPE={mape:.4f} sum_ratio={sr:.4f}")
lo = np.abs(pred_it - y_dv)
print(f"  abs token error: median={np.median(lo):.1f} p90={np.percentile(lo,90):.1f} "
      f"max={lo.max():.0f}")

# corr of input tokens across models
print(f"  corr itok light/mid = {np.corrcoef(dv.itok[:,0], dv.itok[:,1])[0,1]:.5f}, "
      f"light/k1 = {np.corrcoef(dv.itok[:,0], dv.itok[:,2])[0,1]:.5f}")

print()
print("=== (c) family-constant instruction offset (itok - our-text token proxy) ===")
# proxy tokenizer: crude GPT-ish estimate = ceil(bytes/4) is bad for Korean;
# use per-family fitted slope instead and report the RESIDUAL constant.
for f in fams:
    m = np.array([g == f for g in fam_dv])
    if m.sum() < 5:
        continue
    x = np.array([len(t) for t in dv.texts])[m]
    y = y_dv[m]
    # fit y = a*x + b within family; report b (the constant offset) and residual sd
    A = np.column_stack([x, np.ones(len(x))])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ c
    print(f"  {f:16s} n={m.sum():4d} slope={c[0]:.4f} tok/char  intercept={c[1]:8.1f} "
          f"resid sd={res.std():7.1f}  itok/char med={np.median(y/x):.4f}")
