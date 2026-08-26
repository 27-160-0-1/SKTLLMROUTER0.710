# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a12 P1 -- candidate-model profile: where each model wins / loses.

Slices: family x prompt-length bucket x language x num_generations.
Outputs conditional upgrade probabilities and the cost of each upgrade.
Everything printed here is recomputed from data/ only.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from labdata import load_split, MODEL_IDS  # noqa: E402
from ossp_router.similarity import classify_family  # noqa: E402

np.set_printoptions(suppress=True, linewidth=200)

tr, dv = load_split("train"), load_split("dev")


def enrich(sp):
    fam = np.array([classify_family(t) for t in sp.texts])
    nch = np.array([len(t) for t in sp.texts], dtype=float)
    hangul = np.array([sum(1 for c in t if 0xAC00 <= ord(c) <= 0xD7A3) for t in sp.texts], dtype=float)
    lang = np.where(hangul / np.maximum(nch, 1) > 0.05, "ko", "en")
    itok = sp.itok[:, 0] / sp.ngen[:, 0]          # input tokens per generation
    otok = sp.otok / sp.ngen                       # output tokens per generation, (n,3)
    return dict(fam=fam, nch=nch, lang=lang, itok=itok, otok=otok,
                ngen=sp.ngen[:, 0].astype(int), s=sp.score, c=sp.cost)


TR, DV = enrich(tr), enrich(dv)
ALL = dict(fam=np.concatenate([TR["fam"], DV["fam"]]),
           nch=np.concatenate([TR["nch"], DV["nch"]]),
           lang=np.concatenate([TR["lang"], DV["lang"]]),
           itok=np.concatenate([TR["itok"], DV["itok"]]),
           otok=np.vstack([TR["otok"], DV["otok"]]),
           ngen=np.concatenate([TR["ngen"], DV["ngen"]]),
           s=np.vstack([TR["s"], DV["s"]]),
           c=np.vstack([TR["c"], DV["c"]]))
SPLITS = {"train": TR, "dev": DV, "all": ALL}


def hdr(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


hdr("P1.0  global model profile (train / dev / all)")
print(f"{'split':6s} {'n':>5s} " + " ".join(f"{m+'_s':>12s}" for m in MODEL_IDS)
      + " " + " ".join(f"{m+'_cx':>10s}" for m in MODEL_IDS))
for k, D in SPLITS.items():
    n = len(D["fam"])
    cx = D["c"].mean(0) / D["c"][:, 0].mean()
    print(f"{k:6s} {n:5d} " + " ".join(f"{v:12.4f}" for v in D["s"].mean(0))
          + " " + " ".join(f"{v:10.2f}" for v in cx))

hdr("P1.1  per-family slice (train+dev, n=2640) -- score, cost multiple, upgrade economics")
D = ALL
fams = sorted(set(D["fam"]))
print(f"{'family':16s} {'n':>5s} {'ngen4':>6s} {'s_lgt':>7s} {'s_mid':>7s} {'s_k1':>7s} "
      f"{'d10':>7s} {'d21':>7s} {'d20':>7s} {'c1/c0':>7s} {'c2/c1':>7s} {'c2/c0':>7s} "
      f"{'g10/dc':>8s} {'g21/dc':>8s}")
rows = {}
for f in fams:
    m = D["fam"] == f
    s = D["s"][m]
    c = D["c"][m]
    d10 = (s[:, 1] - s[:, 0]).mean()
    d21 = (s[:, 2] - s[:, 1]).mean()
    d20 = (s[:, 2] - s[:, 0]).mean()
    # efficiency measured in "score per unit of light-cost spent extra"
    e10 = d10 / ((c[:, 1] - c[:, 0]).mean() / c[:, 0].mean())
    e21 = d21 / ((c[:, 2] - c[:, 1]).mean() / c[:, 0].mean())
    rows[f] = dict(n=int(m.sum()), d10=d10, d21=d21, e10=e10, e21=e21)
    print(f"{f:16s} {m.sum():5d} {np.mean(D['ngen'][m]==4):6.2f} "
          f"{s[:,0].mean():7.3f} {s[:,1].mean():7.3f} {s[:,2].mean():7.3f} "
          f"{d10:7.3f} {d21:7.3f} {d20:7.3f} "
          f"{c[:,1].mean()/c[:,0].mean():7.2f} {c[:,2].mean()/c[:,1].mean():7.2f} "
          f"{c[:,2].mean()/c[:,0].mean():7.2f} {e10:8.4f} {e21:8.4f}")
allc = D["c"]
print(f"{'ALL':16s} {len(D['fam']):5d} {np.mean(D['ngen']==4):6.2f} "
      f"{D['s'][:,0].mean():7.3f} {D['s'][:,1].mean():7.3f} {D['s'][:,2].mean():7.3f} "
      f"{(D['s'][:,1]-D['s'][:,0]).mean():7.3f} {(D['s'][:,2]-D['s'][:,1]).mean():7.3f} "
      f"{(D['s'][:,2]-D['s'][:,0]).mean():7.3f} "
      f"{allc[:,1].mean()/allc[:,0].mean():7.2f} {allc[:,2].mean()/allc[:,1].mean():7.2f} "
      f"{allc[:,2].mean()/allc[:,0].mean():7.2f}")

hdr("P1.2  conditional upgrade probabilities per family (train+dev)")
print("P(win)  = P(s_hi > s_lo);  P(loss) = P(s_hi < s_lo);  P(tie) = rest")
print(f"{'family':16s} | {'mid>light':>9s} {'mid<light':>9s} {'mid=light':>9s} "
      f"| {'k1>mid':>7s} {'k1<mid':>7s} {'k1=mid':>7s} | {'k1>light':>8s} {'k1<light':>8s}")
for f in fams + ["ALL"]:
    m = np.ones(len(D["fam"]), bool) if f == "ALL" else (D["fam"] == f)
    s = D["s"][m]
    a = lambda x: float(np.mean(x))
    print(f"{f:16s} | {a(s[:,1]>s[:,0]):9.3f} {a(s[:,1]<s[:,0]):9.3f} {a(s[:,1]==s[:,0]):9.3f} "
          f"| {a(s[:,2]>s[:,1]):7.3f} {a(s[:,2]<s[:,1]):7.3f} {a(s[:,2]==s[:,1]):7.3f} "
          f"| {a(s[:,2]>s[:,0]):8.3f} {a(s[:,2]<s[:,0]):8.3f}")

hdr("P1.3  conditional-on-outcome structure (train+dev), using s>=0.5 as 'correct'")
print("cells: P(hi ok | lo ok), P(hi ok | lo fail)  -- 'rescue rate' is the 2nd number")
print(f"{'family':16s} {'n':>5s} | {'P(m|l+)':>8s} {'P(m|l-)':>8s} {'P(k|m+)':>8s} {'P(k|m-)':>8s} "
      f"{'P(k|l-)':>8s} | {'l- frac':>8s} {'m- frac':>8s}")
ok = D["s"] >= 0.5
for f in fams + ["ALL"]:
    m = np.ones(len(D["fam"]), bool) if f == "ALL" else (D["fam"] == f)
    o = ok[m]
    def cp(hi, lo, val):
        sub = o[:, lo] == val
        return float(o[sub, hi].mean()) if sub.sum() else np.nan
    print(f"{f:16s} {m.sum():5d} | {cp(1,0,True):8.3f} {cp(1,0,False):8.3f} "
          f"{cp(2,1,True):8.3f} {cp(2,1,False):8.3f} {cp(2,0,False):8.3f} | "
          f"{1-o[:,0].mean():8.3f} {1-o[:,1].mean():8.3f}")

hdr("P1.4  prompt-length buckets (input tokens per generation, quintiles within train+dev)")
qs = np.quantile(D["itok"], [0.2, 0.4, 0.6, 0.8])
bucket = np.digitize(D["itok"], qs)
print(f"{'bucket':10s} {'itok_rng':>18s} {'n':>5s} {'s_lgt':>7s} {'s_mid':>7s} {'s_k1':>7s} "
      f"{'d10':>7s} {'d21':>7s} {'c1/c0':>7s} {'c2/c0':>7s} {'g10/dc':>8s} {'g21/dc':>8s} {'top fam':>28s}")
for b in range(5):
    m = bucket == b
    s, c = D["s"][m], D["c"][m]
    lo, hi = D["itok"][m].min(), D["itok"][m].max()
    d10 = (s[:, 1] - s[:, 0]).mean(); d21 = (s[:, 2] - s[:, 1]).mean()
    e10 = d10 / ((c[:, 1] - c[:, 0]).mean() / c[:, 0].mean())
    e21 = d21 / ((c[:, 2] - c[:, 1]).mean() / c[:, 0].mean())
    fs, ct = np.unique(D["fam"][m], return_counts=True)
    top = ",".join(f"{a}:{b_}" for a, b_ in sorted(zip(fs, ct), key=lambda z: -z[1])[:2])
    print(f"Q{b+1:<9d} {f'{lo:.0f}-{hi:.0f}':>18s} {m.sum():5d} "
          f"{s[:,0].mean():7.3f} {s[:,1].mean():7.3f} {s[:,2].mean():7.3f} {d10:7.3f} {d21:7.3f} "
          f"{c[:,1].mean()/c[:,0].mean():7.2f} {c[:,2].mean()/c[:,0].mean():7.2f} "
          f"{e10:8.4f} {e21:8.4f} {top:>28s}")

hdr("P1.5  length buckets WITHIN family (does length matter beyond family?)")
print(f"{'family':16s} {'half':>5s} {'n':>5s} {'itok_med':>9s} {'s_lgt':>7s} {'s_mid':>7s} {'s_k1':>7s} {'d10':>7s} {'d21':>7s}")
for f in fams:
    m = D["fam"] == f
    if m.sum() < 40:
        continue
    med = np.median(D["itok"][m])
    for half, sel in (("short", m & (D["itok"] <= med)), ("long", m & (D["itok"] > med))):
        s = D["s"][sel]
        print(f"{f:16s} {half:>5s} {sel.sum():5d} {np.median(D['itok'][sel]):9.0f} "
              f"{s[:,0].mean():7.3f} {s[:,1].mean():7.3f} {s[:,2].mean():7.3f} "
              f"{(s[:,1]-s[:,0]).mean():7.3f} {(s[:,2]-s[:,1]).mean():7.3f}")

hdr("P1.6  language (>=5% hangul chars = ko)")
print(f"{'lang':6s} {'n':>5s} {'s_lgt':>7s} {'s_mid':>7s} {'s_k1':>7s} {'d10':>7s} {'d21':>7s} {'c2/c0':>7s} {'families'}")
for L in ("en", "ko"):
    m = D["lang"] == L
    s, c = D["s"][m], D["c"][m]
    fs, ct = np.unique(D["fam"][m], return_counts=True)
    top = ",".join(f"{a}:{b_}" for a, b_ in sorted(zip(fs, ct), key=lambda z: -z[1]))
    print(f"{L:6s} {m.sum():5d} {s[:,0].mean():7.3f} {s[:,1].mean():7.3f} {s[:,2].mean():7.3f} "
          f"{(s[:,1]-s[:,0]).mean():7.3f} {(s[:,2]-s[:,1]).mean():7.3f} "
          f"{c[:,2].mean()/c[:,0].mean():7.2f} {top}")
# language within the two mixed families
for f in fams:
    m = D["fam"] == f
    if len(set(D["lang"][m])) < 2:
        continue
    for L in ("en", "ko"):
        sel = m & (D["lang"] == L)
        if sel.sum() < 10:
            continue
        s = D["s"][sel]
        print(f"   {f:14s} {L} n={sel.sum():4d} s={s[:,0].mean():.3f}/{s[:,1].mean():.3f}/{s[:,2].mean():.3f}")

hdr("P1.7  num_generations")
print(f"{'ngen':>5s} {'n':>5s} {'s_lgt':>7s} {'s_mid':>7s} {'s_k1':>7s} {'d10':>7s} {'d21':>7s} {'families'}")
for g in (2, 4):
    m = D["ngen"] == g
    s = D["s"][m]
    fs, ct = np.unique(D["fam"][m], return_counts=True)
    top = ",".join(f"{a}:{b_}" for a, b_ in sorted(zip(fs, ct), key=lambda z: -z[1])[:4])
    print(f"{g:5d} {m.sum():5d} {s[:,0].mean():7.3f} {s[:,1].mean():7.3f} {s[:,2].mean():7.3f} "
          f"{(s[:,1]-s[:,0]).mean():7.3f} {(s[:,2]-s[:,1]).mean():7.3f} {top}")
print("\n gsm8k_or_other split by ngen (the only family with both):")
for g in (2, 4):
    m = (D["fam"] == "gsm8k_or_other") & (D["ngen"] == g)
    if m.sum() == 0:
        continue
    s = D["s"][m]
    print(f"   ngen={g} n={m.sum():4d} itok_med={np.median(D['itok'][m]):6.0f} "
          f"s={s[:,0].mean():.3f}/{s[:,1].mean():.3f}/{s[:,2].mean():.3f} "
          f"d10={(s[:,1]-s[:,0]).mean():+.3f} d21={(s[:,2]-s[:,1]).mean():+.3f}")

hdr("P1.8  UPGRADE PROFILE ranked by efficiency (gain per extra light-cost unit), train+dev")
print("light->mid")
for f, r in sorted(rows.items(), key=lambda kv: -kv[1]["e10"]):
    print(f"   {f:16s} n={r['n']:5d} gain={r['d10']:+.3f} eff={r['e10']:8.4f}")
print("mid->k1")
for f, r in sorted(rows.items(), key=lambda kv: -kv[1]["e21"]):
    print(f"   {f:16s} n={r['n']:5d} gain={r['d21']:+.3f} eff={r['e21']:8.4f}")

hdr("P1.9  train-vs-dev stability of the per-family gains (is the profile a stable object?)")
print(f"{'family':16s} {'n_tr':>5s} {'n_dv':>5s} | {'d10_tr':>7s} {'d10_dv':>7s} {'se':>6s} | "
      f"{'d21_tr':>7s} {'d21_dv':>7s} {'se':>6s}")
for f in fams:
    mt, md = TR["fam"] == f, DV["fam"] == f
    if mt.sum() < 5 or md.sum() < 5:
        continue
    st, sd = TR["s"][mt], DV["s"][md]
    a10t, a10d = (st[:, 1] - st[:, 0]), (sd[:, 1] - sd[:, 0])
    a21t, a21d = (st[:, 2] - st[:, 1]), (sd[:, 2] - sd[:, 1])
    se10 = np.sqrt(a10t.var(ddof=1) / mt.sum() + a10d.var(ddof=1) / md.sum())
    se21 = np.sqrt(a21t.var(ddof=1) / mt.sum() + a21d.var(ddof=1) / md.sum())
    print(f"{f:16s} {mt.sum():5d} {md.sum():5d} | {a10t.mean():+7.3f} {a10d.mean():+7.3f} {se10:6.3f} | "
          f"{a21t.mean():+7.3f} {a21d.mean():+7.3f} {se21:6.3f}")
