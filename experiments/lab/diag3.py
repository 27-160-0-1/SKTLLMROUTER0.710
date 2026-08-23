# SPDX-License-Identifier: Apache-2.0
"""(a) is num_generations predictable?  (b) how exactly can input tokens be predicted?
   (c) rough noise-inflation of the oracle ceiling."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_WEIGHT, tier_result
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

tr, dv = load_split("train"), load_split("dev")

print("=== (a) num_generations by family ===")
for sp in (tr, dv):
    fam = np.array([classify_family(t) for t in sp.texts])
    print(f" {sp.name}")
    for f_ in sorted(set(fam)):
        m = fam == f_
        g = sp.ngen[m, 0]
        print(f"   {f_:15s} n={m.sum():4d} ngen4_frac={np.mean(g==4):.3f}")

print("\n=== (b) input-token predictability from cheap text stats ===")
def stats(texts):
    out = []
    for t in texts:
        n_ascii = sum(1 for ch in t if ord(ch) < 128)
        n_hangul = sum(1 for ch in t if 0xAC00 <= ord(ch) <= 0xD7A3)
        n_cjk = sum(1 for ch in t if 0x4E00 <= ord(ch) <= 0x9FFF)
        n_space = t.count(" ") + t.count("\n")
        n_digit = sum(ch.isdigit() for ch in t)
        n_punct = sum(1 for ch in t if not ch.isalnum() and not ch.isspace() and ord(ch) < 128)
        n_other = len(t) - n_ascii - n_hangul - n_cjk
        out.append([len(t), n_ascii, n_hangul, n_cjk, n_space, n_digit, n_punct, n_other,
                    len(t.split()), t.count("\n"), 1.0])
    return np.asarray(out, dtype=float)

Xtr, Xdv = stats(tr.texts), stats(dv.texts)
ytr = tr.itok[:, 0] / tr.ngen[:, 0]
ydv = dv.itok[:, 0] / dv.ngen[:, 0]
famtr = np.array([classify_family(t) for t in tr.texts])
famdv = np.array([classify_family(t) for t in dv.texts])
fams = sorted(set(famtr) | set(famdv))
Ftr = np.stack([(famtr == f).astype(float) for f in fams], 1)
Fdv = np.stack([(famdv == f).astype(float) for f in fams], 1)
for label, A, B in (("stats", Xtr, Xdv), ("stats+family", np.hstack([Xtr, Ftr]), np.hstack([Xdv, Fdv]))):
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    p = B @ w
    rel = np.abs(p - ydv) / np.maximum(ydv, 1)
    print(f"  {label:14s} dev  R2={1-((p-ydv)**2).sum()/((ydv-ydv.mean())**2).sum():.5f}"
          f"  medAPE={np.median(rel)*100:.2f}%  p90APE={np.percentile(rel,90)*100:.2f}%"
          f"  sum_ratio={p.sum()/ydv.sum():.4f}")

print("\n=== (c) oracle noise inflation (simulated) ===")
# assume observed s = Binom(ngen, p)/ngen with p = kNN-free smooth proxy:
# use per-family per-model mean as p, resample, and compare oracle-on-resample vs oracle-on-p
rng = np.random.default_rng(0)
for sp, fam in ((dv, famdv),):
    p = np.zeros_like(sp.score)
    for f_ in set(fam):
        m = fam == f_
        p[m] = sp.score[m].mean(0)
    def wsum(sc):
        return sum(TIER_WEIGHT[t] * tier_result(sc, sp.cost, sp, t, 1.0)["tier_score"] for t in TIERS)
    # oracle that knows p exactly, evaluated on the real noisy outcome
    print(f"  family-mean-p oracle (eval on real s): {wsum(p):.4f}")
    print(f"  true-s oracle:                          {wsum(sp.score):.4f}")
    # simulate: draw fake s from p, run oracle on fake s, evaluate on an independent fake s
    tot_in, tot_out = [], []
    for _ in range(20):
        n = sp.ngen
        s1 = rng.binomial(n.astype(int), p) / n
        s2 = rng.binomial(n.astype(int), p) / n
        a = sum(TIER_WEIGHT[t] * tier_result(s1, sp.cost, sp, t, 1.0)["tier_score"] for t in TIERS)
        # evaluate the s1-oracle allocation on the independent draw s2
        b = 0.0
        for t in TIERS:
            r = tier_result(s1, sp.cost, sp, t, 1.0)
            sc = s2[np.arange(len(sp)), r["sel"]].mean()
            b += TIER_WEIGHT[t] * (sc if r["passed"] else 0.0)
        tot_in.append(a); tot_out.append(b)
    print(f"  simulated: oracle-on-own-draw={np.mean(tot_in):.4f}  same alloc on fresh draw={np.mean(tot_out):.4f}"
          f"  inflation={np.mean(tot_in)-np.mean(tot_out):.4f}")
