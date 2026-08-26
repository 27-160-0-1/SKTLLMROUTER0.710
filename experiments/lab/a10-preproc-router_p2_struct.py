# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""a10 P2 — is the per-item cost RATIO intrinsically easier to predict than the level?

Cost algebra (exact): c_m = rate_m * (I_m + 4*O_m) / 1e6, rate = 1 / 2.127 / 6.565.
Define effective tokens T_m = I_m + 4*O_m.  Then

    r_m = c_m/c_0 = rate_m * T_m / T_0
    w_i = c_i0 / sum_j c_j0 = T_i0 / sum_j T_j0

So the whole allocation problem depends on the cost side only through T.
Question: how much of the variance of log T_m is SHARED across models?  If the
residuals correlate strongly, log(T_m/T_0) is a much lower-variance target than
log T_m, and a direct ratio head beats the current difference-of-two-heads.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

ROOT = Path(__file__).resolve().parents[2]
tr, dv = load_split("train"), load_split("dev")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)

print("=" * 78)
print("(0) is num_generations model-invariant within an episode?")
for sp in (tr, dv):
    same = np.all(sp.ngen == sp.ngen[:, :1])
    print(f"    {sp.name}: identical across the 3 models for every episode -> {bool(same)}"
          f"   values={sorted(set(sp.ngen[:,0].tolist()))}")
print("    (if True, the ngen factor 2-vs-4 cancels exactly in every ratio c_m/c_0)")

print()
print("=" * 78)
print("(1) input tokens: how close are I_m across models?")
for sp in (tr, dv):
    for j in (1, 2):
        q = sp.itok[:, j] / sp.itok[:, 0]
        print(f"    {sp.name}: I_{MODEL_IDS[j]}/I_light  mean={q.mean():.4f} sd={q.std():.4f} "
              f"p1={np.percentile(q,1):.4f} p99={np.percentile(q,99):.4f}")

print()
print("=" * 78)
print("(2) variance structure of log T, after removing a family-mean model")
print("    (family means fitted on TRAIN, residuals measured on DEV)")
famtr = np.array([classify_family(t) for t in tr.texts])
famdv = np.array([classify_family(t) for t in dv.texts])
fams = sorted(set(famtr) | set(famdv))
Ttr = tr.itok + 4.0 * tr.otok
Tdv = dv.itok + 4.0 * dv.otok
lTtr, lTdv = np.log(Ttr), np.log(Tdv)

mu = np.zeros((len(fams), 3))
for k, f in enumerate(fams):
    m = famtr == f
    mu[k] = lTtr[m].mean(0)
fidx = np.array([fams.index(f) for f in famdv])
res = lTdv - mu[fidx]          # (880,3) residual of log effective tokens
print("    residual sd of log T per model      :", np.round(res.std(0), 4))
Cres = np.corrcoef(res.T)
print("    residual CORRELATION across models:")
for j, m in enumerate(MODEL_IDS):
    print(f"      {m:11s} " + " ".join(f"{Cres[j,k]:+.3f}" for k in range(3)))
for j in (1, 2):
    dres = res[:, j] - res[:, 0]
    print(f"    log(T_{MODEL_IDS[j]}/T_light) residual sd = {dres.std():.4f}   "
          f"vs level sd {res[:,j].std():.4f}   "
          f"(independent-errors would give {np.hypot(res[:,j].std(), res[:,0].std()):.4f})")

print()
print("=" * 78)
print("(3) same thing but with a strong feature model (GBM on cheap text stats + family)")


def feats(sp, fam):
    out = []
    for t in sp.texts:
        n_ascii = sum(1 for ch in t if ord(ch) < 128)
        n_hangul = sum(1 for ch in t if 0xAC00 <= ord(ch) <= 0xD7A3)
        n_space = t.count(" ") + t.count("\n")
        n_digit = sum(ch.isdigit() for ch in t)
        n_punct = sum(1 for ch in t if not ch.isalnum() and not ch.isspace() and ord(ch) < 128)
        out.append([len(t), n_ascii, n_hangul, n_space, n_digit, n_punct,
                    len(t.split()), t.count("\n"), t.count("?"), t.count("="),
                    np.log1p(len(t))])
    X = np.asarray(out, float)
    F = np.stack([(fam == f).astype(float) for f in fams], 1)
    return np.hstack([X, F])


Xtr, Xdv = feats(tr, famtr), feats(dv, famdv)
# ngen is a legitimate runtime-unknown, but it is family-determined; include the
# TRAIN family frequency of ngen==4 as a feature (allowed: derived from train labels)
p4 = np.array([float((tr.ngen[famtr == f, 0] == 4).mean()) for f in fams])
Xtr = np.hstack([Xtr, p4[[fams.index(f) for f in famtr]][:, None]])
Xdv = np.hstack([Xdv, p4[fidx][:, None]])

from sklearn.ensemble import HistGradientBoostingRegressor

PAR = dict(learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=30,
           l2_regularization=3.0, max_iter=400, early_stopping=True,
           validation_fraction=0.15, random_state=7)


def fit_pred(y):
    g = HistGradientBoostingRegressor(**PAR).fit(Xtr, y)
    return g.predict(Xdv), g.predict(Xtr)


# parameterisation A: three independent log T heads
pA = np.stack([fit_pred(lTtr[:, j])[0] for j in range(3)], 1)
# parameterisation B: log T_0 head + two DIRECT log-ratio heads
p0 = fit_pred(lTtr[:, 0])[0]
pr1 = fit_pred(lTtr[:, 1] - lTtr[:, 0])[0]
pr2 = fit_pred(lTtr[:, 2] - lTtr[:, 0])[0]
pB = np.stack([p0, p0 + pr1, p0 + pr2], 1)
# parameterisation C: shared input-token head + per-model output head
lItr = np.log(tr.itok[:, 0])
lOtr = np.log(np.maximum(tr.otok, 1.0))
pI = fit_pred(lItr)[0]
pO = np.stack([fit_pred(lOtr[:, j])[0] for j in range(3)], 1)
pC = np.log(np.exp(pI)[:, None] + 4.0 * np.exp(pO))

print(f"    {'param':34s} {'sd log T err':>28s} | {'sd log r err (mid / k1)':>26s}")
for nm, pp in (("A independent log-T heads", pA),
               ("B log-T0 + direct log-ratio", pB),
               ("C shared input + per-model out", pC)):
    e = pp - lTdv
    e1 = (pp[:, 1] - pp[:, 0]) - (lTdv[:, 1] - lTdv[:, 0])
    e2 = (pp[:, 2] - pp[:, 0]) - (lTdv[:, 2] - lTdv[:, 0])
    print(f"    {nm:34s} {np.round(e.std(0),4)!s:>28s} | "
          f"{e1.std():.4f} / {e2.std():.4f}")
edep = np.log(P['cost_premium']) - np.log(dv.cost)
e1d = (np.log(P['cost_premium'][:, 1]/P['cost_premium'][:, 0])
       - np.log(dv.cost[:, 1]/dv.cost[:, 0]))
e2d = (np.log(P['cost_premium'][:, 2]/P['cost_premium'][:, 0])
       - np.log(dv.cost[:, 2]/dv.cost[:, 0]))
print(f"    {'(deployed E43 cost head, premium)':34s} {np.round(edep.std(0),4)!s:>28s} | "
      f"{e1d.std():.4f} / {e2d.std():.4f}")

print()
print("=" * 78)
print("(4) ORACLE bound: how well could r be predicted if we knew T_0 exactly?")
print("    residual sd of log(T_m/T_0) around the family mean, DEV in-sample:")
for j in (1, 2):
    d = lTdv[:, j] - lTdv[:, 0]
    lev = lTdv[:, j]
    sd_d, sd_l = [], []
    for f in fams:
        m = famdv == f
        if m.sum() < 5:
            continue
        sd_d.append(d[m].std() * np.sqrt(m.sum()))
        sd_l.append(lev[m].std() * np.sqrt(m.sum()))
    sd_d = np.sqrt(np.sum(np.square(sd_d)) / len(dv))
    sd_l = np.sqrt(np.sum(np.square(sd_l)) / len(dv))
    print(f"      {MODEL_IDS[j]:11s} within-family sd:  ratio {sd_d:.4f}   level {sd_l:.4f}"
          f"   -> ratio/level = {sd_d/sd_l:.3f}")

OUT = Path(r"C:\Users\PJ05\AppData\Local\Temp\claude\C--portable-skt-LLM1-LLM-ROUTE-0-7000"
           r"\377d7fd8-9983-4bec-bb08-163cc405f7a3\scratchpad\a10_p2_costpreds.npz")
OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(OUT, pA=pA, pB=pB, pC=pC, lTdv=lTdv, fidx=fidx, fams=np.array(fams))
print("\nwrote", OUT)
