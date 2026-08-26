# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""a10 P5 - the real mechanism behind 'no k1 for code'.

P4 found that removing axk1-think from the action set of the `code` family
raises the bootstrap EV by +0.0044 and lets premium safety go 0.72 -> 0.90.
Hypothesis: it is not that code-k1 is a bad buy per se, it is that the log-cost
head reports exp(E[log c]) which under-states E[c] by exp(sigma^2/2), and
sigma is by far the largest for code-k1 -> the allocator believes code-k1 is
~1.7x cheaper than it is and over-buys it.

If that is the mechanism, a per-(family, model) sum-matched smearing factor
estimated on TRAIN ONLY, applied to the UTILITY cost (not just the ledger),
should reproduce most of the gain without a hand-picked clamp.

Everything here is train-derived: the factors come from out-of-fold predictions
of a GBM cost head fitted on train only; they are then applied to the deployed
E43 dev predictions.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from labdata import load_split, MODEL_IDS, TIERS, TIER_MULT, TIER_WEIGHT
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ossp_router.similarity import classify_family

ROOT = Path(__file__).resolve().parents[2]
dv = load_split("dev")
tr = load_split("train")
P = np.load(ROOT / "reports/lab/dev_preds_e43.npz", allow_pickle=True)
n = len(dv)
ar = np.arange(n)
famdv = np.array([classify_family(t) for t in dv.texts])
famtr = np.array([classify_family(t) for t in tr.texts])
fams = sorted(set(famtr) | set(famdv))
FI = {f: i for i, f in enumerate(fams)}
fid = np.array([FI[f] for f in famdv])
fidtr = np.array([FI[f] for f in famtr])

print("=" * 92)
print("(1) is the code/k1 story present in TRAIN as well?  (pure data facts)")
print(f"{'family':14s} | {'train n':>7s} {'sc L/M/K':>20s} {'costratio M/K':>14s} "
      f"{'sd log c k1':>11s} {'exp(s2/2) k1':>12s}")
for k, f in enumerate(fams):
    mt = fidtr == k
    cs = tr.cost[mt].sum(0) / tr.cost[mt, 0].sum()
    sd = np.log(tr.cost[mt, 2]).std()
    print(f"{f:14s} | {int(mt.sum()):7d} "
          f"{np.round(tr.score[mt].mean(0),3)!s:>20s} {np.round(cs[1:],1)!s:>14s} "
          f"{sd:11.3f} {np.exp(sd**2/2):12.2f}")

print()
print("    efficiency of the k1 upgrade per family (train), in score per unit of")
print("    'light-total' budget:  sum(s_k1-s_mid) / sum(c_k1-c_mid) * mean(c_light)")
for split, nm in ((tr, "train"), (dv, "dev")):
    fidx = fidtr if nm == "train" else fid
    row = []
    for k, f in enumerate(fams):
        m = fidx == k
        dsc = (split.score[m, 2] - split.score[m, 1]).sum()
        dc = (split.cost[m, 2] - split.cost[m, 1]).sum() / split.cost[:, 0].sum()
        row.append(f"{f[:6]}={dsc/len(split)/max(dc,1e-9):6.3f}")
    print(f"      {nm:5s} " + " ".join(row))
print("    (same for the mid upgrade)")
for split, nm in ((tr, "train"), (dv, "dev")):
    fidx = fidtr if nm == "train" else fid
    row = []
    for k, f in enumerate(fams):
        m = fidx == k
        dsc = (split.score[m, 1] - split.score[m, 0]).sum()
        dc = (split.cost[m, 1] - split.cost[m, 0]).sum() / split.cost[:, 0].sum()
        row.append(f"{f[:6]}={dsc/len(split)/max(dc,1e-9):6.3f}")
    print(f"      {nm:5s} " + " ".join(row))

# --------------------------------------------------------------- train heads
print()
print("=" * 92)
print("(2) train-only OOF cost head -> per-(family, model) sum-matched smearing factors")


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


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

Xtr = feats(tr, famtr)
lc_tr = np.log(tr.cost)
PAR = dict(learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=30,
           l2_regularization=3.0, max_iter=400, early_stopping=True,
           validation_fraction=0.15, random_state=7)
oof = np.zeros_like(lc_tr)
kf = KFold(5, shuffle=True, random_state=7)
for a_, b_ in kf.split(Xtr):
    for j in range(3):
        g = HistGradientBoostingRegressor(**PAR).fit(Xtr[a_], lc_tr[a_, j])
        oof[b_, j] = g.predict(Xtr[b_])
pred_tr = np.exp(oof)
print("    train OOF log-cost residual sd:", np.round((oof - lc_tr).std(0), 4))
FAC = np.ones((len(fams), 3))
for k, f in enumerate(fams):
    m = fidtr == k
    FAC[k] = tr.cost[m].sum(0) / pred_tr[m].sum(0)
GFAC = tr.cost.sum(0) / pred_tr.sum(0)
print("    global smearing factor (train):", np.round(GFAC, 3))
print(f"    {'family':14s} " + " ".join(f"{m[:9]:>9s}" for m in MODEL_IDS)
      + "   | dev true/pred of the DEPLOYED head")
for k, f in enumerate(fams):
    m = fid == k
    devfac = dv.cost[m].sum(0) / P["cost_premium"][m].sum(0)
    print(f"    {f:14s} " + " ".join(f"{FAC[k,j]:9.3f}" for j in range(3))
          + "   | " + " ".join(f"{devfac[j]:6.3f}" for j in range(3)))

# ------------------------------------------------------------------ harness
def allocate2(pred_score, util_cost, ledger_cost, mult, safety, mask=None):
    N = len(pred_score); a = np.arange(N)
    cap = ledger_cost[:, 0].sum() * max(1.0, mult * safety)
    U0 = pred_score if mask is None else np.where(mask, pred_score, -1e18)
    denom = util_cost[:, 0].sum()

    def choose(pen):
        return (U0 - pen * util_cost / denom).argmax(axis=1)

    sel = choose(0.0); tot = ledger_cost[a, sel].sum()
    if tot > cap:
        low, high = 0.0, 1.0
        sel = choose(high); tot = ledger_cost[a, sel].sum()
        while tot > cap and high < 2 ** 60:
            low, high = high, high * 2.0
            sel = choose(high); tot = ledger_cost[a, sel].sum()
        for _ in range(40):
            mid = (low + high) / 2.0
            cand = choose(mid); ct = ledger_cost[a, cand].sum()
            if ct <= cap:
                high, sel, tot = mid, cand, ct
            else:
                low = mid
    if tot > cap:
        sel = np.zeros(N, dtype=int)
    return sel


GRID = np.round(np.arange(0.62, 1.121, 0.02), 3)


def evaluate(cfg, batches, grid=GRID):
    out = {}
    for t in TIERS:
        s, uc, lc, mk = cfg(t)
        mult = TIER_MULT[t]
        best = None
        for sf in grid:
            vals = np.empty(len(batches)); nb = 0
            for bi, rows in enumerate(batches):
                sel = allocate2(s[rows], uc[rows], lc[rows], mult, float(sf),
                                None if mk is None else mk[rows])
                tc = dv.cost[rows]
                ok = tc[ar, sel].sum() / tc[:, 0].sum() <= mult + 1e-15
                vals[bi] = dv.score[rows][ar, sel].mean() if ok else 0.0
                nb += 0 if ok else 1
            ev = float(vals.mean())
            if best is None or ev > best[1]:
                best = (float(sf), ev, nb / len(batches))
        sel = allocate2(s, uc, lc, mult, best[0], mk)
        ratio = dv.cost[ar, sel].sum() / dv.cost[:, 0].sum()
        ok = ratio <= mult + 1e-15
        out[t] = dict(safety=best[0], ev=best[1], bust=best[2],
                      dev_score=float(dv.score[ar, sel].mean()),
                      dev_ratio=float(ratio), dev_ok=ok,
                      nk1=int((sel == 2).sum()), nmid=int((sel == 1).sum()))
    out["ev"] = sum(TIER_WEIGHT[t] * out[t]["ev"] for t in TIERS)
    out["dev"] = sum(TIER_WEIGHT[t] * (out[t]["dev_score"] if out[t]["dev_ok"] else 0.0)
                     for t in TIERS)
    return out


def base(t):
    return P[f"score_{t}"], P[f"cost_{t}"], P[f"cost_{t}"], None


def mk_smear(where, mode="family"):
    f = FAC if mode == "family" else np.tile(GFAC, (len(fams), 1))

    def cfg(t):
        pc = P[f"cost_{t}"]
        sm = pc * f[fid]
        if where == "ledger":
            return P[f"score_{t}"], pc, sm, None
        if where == "util":
            return P[f"score_{t}"], sm, pc, None
        return P[f"score_{t}"], sm, sm, None
    return cfg


def mk_smear_k1only(where):
    def cfg(t):
        pc = P[f"cost_{t}"]
        f = np.ones((len(fams), 3)); f[:, 2] = FAC[:, 2]
        sm = pc * f[fid]
        if where == "ledger":
            return P[f"score_{t}"], pc, sm, None
        if where == "util":
            return P[f"score_{t}"], sm, pc, None
        return P[f"score_{t}"], sm, sm, None
    return cfg


def mk_nok1(families):
    def cfg(t):
        mask = np.ones((n, 3), bool)
        for f in families:
            mask[fid == FI[f], 2] = False
        return P[f"score_{t}"], P[f"cost_{t}"], P[f"cost_{t}"], mask
    return cfg


CONFIGS = [
    ("base (deployed inputs)", base),
    ("smear BOTH, family x model (train)", mk_smear("both")),
    ("smear UTILITY only, family x model", mk_smear("util")),
    ("smear LEDGER only, family x model", mk_smear("ledger")),
    ("smear BOTH, global factor only", mk_smear("both", "global")),
    ("smear BOTH, k1 column only", mk_smear_k1only("both")),
    ("hard: no k1 for code", mk_nok1(["code"])),
    ("smear BOTH + no k1 for code",
     lambda t: (P[f"score_{t}"], P[f"cost_{t}"] * FAC[fid], P[f"cost_{t}"] * FAC[fid],
                np.where(np.arange(3)[None, :] == 2,
                         (fid != FI["code"])[:, None], True))),
]

for SEED in (7, 17, 23):
    rng = np.random.default_rng(SEED)
    batches = [rng.integers(0, n, n) for _ in range(250)]
    print()
    print("=" * 92)
    print(f"(3) paired bootstrap seed={SEED} B=250, EV-optimal safety per config")
    print(f"{'config':38s} {'EV':>7s} {'dEV':>7s} {'dev':>7s}  {'safety f/b/p':>14s}"
          f"  {'bust f/b/p':>14s}  {'prem k1/mid':>11s}")
    ref = None
    for name, cfg in CONFIGS:
        r = evaluate(cfg, batches)
        if ref is None:
            ref = r["ev"]
        print(f"{name:38s} {r['ev']:7.4f} {r['ev']-ref:+7.4f} {r['dev']:7.4f}  "
              + "/".join(f"{r[t]['safety']:.2f}" for t in TIERS) + "  "
              + "/".join(f"{r[t]['bust']:.3f}" for t in TIERS)
              + f"  {r['premium']['nk1']:4d}/{r['premium']['nmid']:4d}"
              + "  tiers " + " ".join(f"{r[t]['dev_score']:.4f}" for t in TIERS))
