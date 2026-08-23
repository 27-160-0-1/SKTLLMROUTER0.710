# SPDX-License-Identifier: Apache-2.0
"""a12 P2 -- axk1-think as a reasoning object: output-length law, truncation, cost.

Q1 how does k1 output length depend on item type?
Q2 does it "think longer when it is failing"?
Q3 does length saturate at a context limit, how often, and what does that do to score?
Q4 output-length asymmetry vs the two non-reasoning models.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from labdata import load_split  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

tr, dv = load_split("train"), load_split("dev")
fam = np.concatenate([[classify_family(t) for t in tr.texts], [classify_family(t) for t in dv.texts]])
S = np.vstack([tr.score, dv.score])
C = np.vstack([tr.cost, dv.cost])
IT = np.vstack([tr.itok, dv.itok])
OT = np.vstack([tr.otok, dv.otok])
NG = np.vstack([tr.ngen, dv.ngen])
split = np.array(["train"] * len(tr.texts) + ["dev"] * len(dv.texts))
og = OT / NG          # output tokens per generation
ig = IT / NG          # input tokens per generation
fams = sorted(set(fam))


def hdr(t):
    print("\n" + "=" * 104)
    print(t)
    print("=" * 104)


hdr("P2.0  output tokens per generation -- distribution per model")
for j, m in enumerate(("light", "mid", "k1")):
    q = np.percentile(og[:, j], [10, 25, 50, 75, 90, 95, 99, 100])
    print(f"{m:6s} mean={og[:,j].mean():8.1f} " + " ".join(f"p{p}={v:8.0f}" for p, v in
          zip([10, 25, 50, 75, 90, 95, 99, 100], q)))
print("\ntotal tokens per generation (in+out), k1:")
tot = ig[:, 2] + og[:, 2]
print("  ", np.round(np.percentile(tot, [50, 90, 99, 99.5, 100]), 0))
print("  n with in+out per gen >= 32000:", int((tot >= 32000).sum()),
      " >= 32768:", int((tot >= 32768).sum()),
      " out per gen >= 32000:", int((og[:, 2] >= 32000).sum()))
print("  max out/gen =", og[:, 2].max(), " max out total =", OT[:, 2].max(),
      " max in/gen =", ig[:, 2].max())

hdr("P2.1  k1 output length by family (per generation)")
print(f"{'family':16s} {'n':>5s} {'mean':>8s} {'p25':>7s} {'p50':>7s} {'p75':>7s} {'p90':>7s} {'p99':>8s} "
      f"{'max':>8s} {'sd(log)':>8s} {'share of k1 cost':>17s}")
tot_k1_cost = C[:, 2].sum()
for f in fams:
    m = fam == f
    v = og[m, 2]
    print(f"{f:16s} {m.sum():5d} {v.mean():8.0f} " +
          " ".join(f"{x:7.0f}" for x in np.percentile(v, [25, 50, 75, 90])) +
          f" {np.percentile(v,99):8.0f} {v.max():8.0f} {np.std(np.log(np.maximum(v,1))):8.3f}"
          f" {C[m,2].sum()/tot_k1_cost:17.3f}")

hdr("P2.2  'thinks longer when failing?'  correlation of log(out/gen) with own score")
print(f"{'family':16s} {'n':>5s} | " + " ".join(f"{m+'_r':>9s}" for m in ("light", "mid", "k1")) +
      " | k1 out/gen by own score bucket (s=0 / 0<s<1 / s=1)")
for f in fams + ["ALL"]:
    m = np.ones(len(fam), bool) if f == "ALL" else (fam == f)
    rs = []
    for j in range(3):
        x = np.log(np.maximum(og[m, j], 1)); y = S[m, j]
        rs.append(np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else np.nan)
    b0 = og[m & (S[:, 2] == 0), 2]
    bm = og[m & (S[:, 2] > 0) & (S[:, 2] < 1), 2]
    b1 = og[m & (S[:, 2] == 1), 2]
    def med(a):
        return f"{np.median(a):6.0f}(n={len(a)})" if len(a) else "     -      "
    print(f"{f:16s} {m.sum():5d} | " + " ".join(f"{r:9.3f}" for r in rs) +
          f" | {med(b0)} {med(bm)} {med(b1)}")

hdr("P2.3  partial correlation: within-family, is length still informative about failure?")
# residualise log length and score on family means, then correlate
lg = np.log(np.maximum(og[:, 2], 1))
lgr = lg.copy(); sr = S[:, 2].copy()
for f in fams:
    m = fam == f
    lgr[m] -= lgr[m].mean(); sr[m] -= sr[m].mean()
print(f"  raw corr(log out_k1, s_k1)         = {np.corrcoef(lg, S[:,2])[0,1]:+.3f}")
print(f"  within-family corr                  = {np.corrcoef(lgr, sr)[0,1]:+.3f}")
print(f"  raw corr(log out_k1, s_light)       = {np.corrcoef(lg, S[:,0])[0,1]:+.3f}")
print(f"  within-family corr with s_light     = {np.corrcoef(lgr, S[:,0]-np.array([S[fam==f,0].mean() for f in fam]))[0,1]:+.3f}")
print(f"  corr(log out_k1, log out_mid)       = {np.corrcoef(lg, np.log(np.maximum(og[:,1],1)))[0,1]:+.3f}")
print(f"  corr(log out_k1, log in_tok)        = {np.corrcoef(lg, np.log(np.maximum(ig[:,2],1)))[0,1]:+.3f}")
print(f"  within-family corr with log in_tok  = "
      f"{np.corrcoef(lgr, np.log(np.maximum(ig[:,2],1))-np.array([np.log(np.maximum(ig[fam==f,2],1)).mean() for f in fam]))[0,1]:+.3f}")

hdr("P2.4  long-generation items: score and cost consequences (per-gen out tokens deciles)")
dec = np.digitize(og[:, 2], np.percentile(og[:, 2], np.arange(10, 100, 10)))
print(f"{'dec':>3s} {'out/gen rng':>16s} {'n':>5s} {'s_k1':>7s} {'s_mid':>7s} {'s_lgt':>7s} {'d21':>7s} "
      f"{'cost/gen $':>10s} {'c2/c0':>7s} {'share k1 cost':>13s} {'top families'}")
for d in range(10):
    m = dec == d
    fs, ct = np.unique(fam[m], return_counts=True)
    top = ",".join(f"{a}:{b}" for a, b in sorted(zip(fs, ct), key=lambda z: -z[1])[:3])
    print(f"{d:3d} {f'{og[m,2].min():.0f}-{og[m,2].max():.0f}':>16s} {m.sum():5d} "
          f"{S[m,2].mean():7.3f} {S[m,1].mean():7.3f} {S[m,0].mean():7.3f} "
          f"{(S[m,2]-S[m,1]).mean():+7.3f} {C[m,2].mean()*1e3:10.4f} "
          f"{C[m,2].mean()/C[m,0].mean():7.1f} {C[m,2].sum()/tot_k1_cost:13.3f} {top}")

hdr("P2.5  truncation: items whose per-gen output is near the 32,768 cap")
for thr in (8000, 16000, 24000, 30000, 32000, 32500):
    m = og[:, 2] >= thr
    if m.sum() == 0:
        continue
    print(f"  out/gen >= {thr:6d}: n={m.sum():4d} ({m.mean()*100:5.2f}%)  s_k1={S[m,2].mean():.3f} "
          f"s_mid={S[m,1].mean():.3f} s_light={S[m,0].mean():.3f}  "
          f"share of total k1 cost={C[m,2].sum()/tot_k1_cost:.3f}  "
          f"fams=" + ",".join(f"{a}:{b}" for a, b in
                              sorted(zip(*np.unique(fam[m], return_counts=True)), key=lambda z: -z[1])[:4]))
print("\n  exact top-20 per-gen output lengths:")
o = np.argsort(-og[:, 2])[:20]
for i in o:
    print(f"    {split[i]:5s} {fam[i]:15s} ngen={int(NG[i,2])} out/gen={og[i,2]:8.1f} in/gen={ig[i,2]:7.0f} "
          f"in+out={ig[i,2]+og[i,2]:8.0f} s_k1={S[i,2]:.2f} s_mid={S[i,1]:.2f} s_lgt={S[i,0]:.2f} "
          f"cost_k1=${C[i,2]*1e3:.3f}m")

hdr("P2.6  what does a very long (probably truncated) generation do to the score?")
print("  P(s_k1 = 0 | out/gen bucket)")
edges = [0, 500, 1000, 2000, 4000, 8000, 16000, 32768]
for a, b in zip(edges[:-1], edges[1:]):
    m = (og[:, 2] >= a) & (og[:, 2] < b)
    if m.sum() == 0:
        continue
    print(f"   [{a:6d},{b:6d}) n={m.sum():5d}  P(s_k1=0)={np.mean(S[m,2]==0):.3f}  "
          f"P(s_k1=1)={np.mean(S[m,2]==1):.3f}  mean s_k1={S[m,2].mean():.3f}  "
          f"mean s_mid={S[m,1].mean():.3f}  k1 beats mid={np.mean(S[m,2]>S[m,1]):.3f} "
          f" loses={np.mean(S[m,2]<S[m,1]):.3f}")

hdr("P2.7  k1 cost concentration -- how much of the budget the tail eats")
o = np.argsort(-C[:, 2])
cum = np.cumsum(C[o, 2]) / C[:, 2].sum()
for k in (10, 26, 53, 132, 264, 528):
    print(f"  top {k:4d} items ({k/len(fam)*100:5.1f}%) = {cum[k-1]*100:5.1f}% of all k1 cost; "
          f"their mean s_k1={S[o[:k],2].mean():.3f} vs mean s_mid={S[o[:k],1].mean():.3f} "
          f"(d21={(S[o[:k],2]-S[o[:k],1]).mean():+.3f})")

hdr("P2.8  is the length law predictable from the prompt?  R2 of family-only and family+len")
y = np.log(np.maximum(og[:, 2], 1))
X0 = np.ones((len(y), 1))
Xf = np.stack([(fam == f).astype(float) for f in fams], 1)
Xl = np.column_stack([Xf, np.log(np.maximum(ig[:, 2], 1)), np.log(np.maximum(ig[:, 2], 1)) ** 2])
for name, X in (("intercept", X0), ("family", Xf), ("family+loglen", Xl)):
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ w
    print(f"  {name:14s} R2={1-r.var()/y.var():.3f}  resid sd(log)={r.std():.3f} (= x{np.exp(r.std()):.2f})")
# add the true light/mid output lengths (unavailable at runtime, an upper reference)
Xo = np.column_stack([Xl, np.log(np.maximum(og[:, 0], 1)), np.log(np.maximum(og[:, 1], 1))])
w, *_ = np.linalg.lstsq(Xo, y, rcond=None)
r = y - Xo @ w
print(f"  {'+true out_l/out_m':14s} R2={1-r.var()/y.var():.3f}  resid sd(log)={r.std():.3f} "
      f"(oracle-ish reference, not runtime-available)")
