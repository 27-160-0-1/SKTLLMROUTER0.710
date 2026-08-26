# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b05 step 11: (a) a decision-focused surrogate that differentiates the
allocator's own outcome, and (b) a second stacking layer.

(a) DFL.  The allocator takes the upper-concave-envelope segments of every item
    sorted by slope Delta_s / Delta_c until the cap binds.  Replace the hard
    prefix by w = sigmoid((log m - log pi)/tau) with pi solved (no grad) so that
    the soft selection spends exactly the cap, enforce the envelope by
    w2 <- w1 * sigmoid(...), and take
        loss = - sum_t weight_t * ( realised score of the soft selection )
               + lambda * hinge( realised spend - cap )
    which is differentiable in the predicted slopes.  The head outputs the
    log-slope directly; the gain handed back to the pipeline is m * Delta_c_pred.

(b) Level-2 stacking.  Inside every outer fold, an inner 5-fold produces
    out-of-fold level-1 predictions on the fit rows; the level-2 head is trained
    on [58 features | level-1 predictions].  This is the "more layers of
    out-of-fold stacking" arm of the brief, at 5x the level-1 training cost.
"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("b05lib", HERE / "b05-time-heavy_lib.py")
lib = importlib.util.module_from_spec(spec); sys.modules["b05lib"] = lib
spec.loader.exec_module(lib)
from harness import DEPLOYED_EXP, DEPLOYED_CFG, TIERS, MULTS, W  # noqa: E402
import bench2 as B  # noqa: E402

EXP = dict(DEPLOYED_EXP, legacy_oof_meta=True)
lab = lib.XLab()
cv, arr, parts = lib.build_stage(lab, EXP, tag="b05base")
POS = {int(v): k for k, v in enumerate(cv["idx"])}
D = lab.delta_targets
TC = lab.true_c
TS = lab.true_s
DC = np.column_stack([TC[:, 1] - TC[:, 0], TC[:, 2] - TC[:, 1]])
GP = dict(max_iter=EXP["gbm_iter"], learning_rate=EXP["gbm_lr"],
          max_leaf_nodes=EXP["gbm_leaves"], min_samples_leaf=EXP["gbm_min_leaf"],
          l2_regularization=EXP["gbm_l2"], early_stopping=True,
          validation_fraction=0.15, random_state=11)
OUT = Path("reports/lab/b05_dfl.json")
ROWS = json.loads(OUT.read_text()) if OUT.exists() else []
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(2)   # the tensors are tiny; thread sync dominates otherwise


def report(name, cvg, devg, secs, cfg=None):
    dg = lib.gain_axis(lab, cv["idx"], cvg[:, 0], cvg[:, 1])
    cfg = dict(DEPLOYED_CFG, **(cfg or {}))
    r = B.run(lab, dict(cv, gain=cvg), dict(arr, gain=devg), cfg, label=name, verbose=False)
    row = dict(name=name, secs=round(secs, 1), cfg={k: cfg[k] for k in ("gain_alpha", "rank_beta")},
               **{k: round(v, 4) for k, v in dg.items()}, EV=round(r["EV"], 6),
               raw=round(sum(W[t] * r["det"][t]["raw"] for t in TIERS), 6),
               dev=round(r["dev"], 6), safety=[round(r["safety"][t], 3) for t in TIERS],
               ratio=[round(r["dev_tiers"][t]["ratio"], 3) for t in TIERS])
    ROWS.append(row); OUT.write_text(json.dumps(ROWS, indent=1), encoding="utf-8")
    print(f"{name:36s}{secs:7.1f}s d1={dg['corr1']:+.4f} d2={dg['corr2']:+.4f} "
          f"A1={dg['auc1']:.4f} A2={dg['auc2']:.4f} EV={row['EV']:.6f} raw={row['raw']:.6f} "
          f"dev={row['dev']:.6f} sf={row['safety']} r={row['ratio']}", flush=True)


# --------------------------------------------------------------------- (a) DFL
class MLP(nn.Module):
    def __init__(self, d, h=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, h), nn.GELU(),
                               nn.Linear(h, 2))

    def forward(self, x):
        return self.f(x)


def soft_pi(u, dc, budget, tau):
    """log-penalty such that the soft selection spends `budget` (no grad).

    Vectorised bisection: all candidate penalties are evaluated at once, so the
    cost is one (n x G) matrix per call instead of G python round-trips.
    """
    lo = float(u.min().item()) - 5.0
    hi = float(u.max().item()) + 5.0
    grid = torch.linspace(lo, hi, 256, device=u.device)
    w1 = torch.sigmoid((u[:, 0:1] - grid[None, :]) / tau)
    w2 = w1 * torch.sigmoid((u[:, 1:2] - grid[None, :]) / tau)
    spend = (w1 * dc[:, 0:1] + w2 * dc[:, 1:2]).sum(0)
    k = int(torch.searchsorted(-spend, torch.tensor(-budget, device=u.device)).item())
    return float(grid[min(max(k, 0), 255)].item())


def _surrogate(u, ds, dc, s0, base, n, tau, lam):
    total = 0.0
    for t in TIERS:
        budget = base * (MULTS[t] - 1.0)
        with torch.no_grad():
            pi = soft_pi(u.detach(), dc, budget, tau)
        w1 = torch.sigmoid((u[:, 0] - pi) / tau)
        w2 = w1 * torch.sigmoid((u[:, 1] - pi) / tau)
        score = (s0 + (w1 * ds[:, 0] + w2 * ds[:, 1]).sum()) / n
        spend = (w1 * dc[:, 0] + w2 * dc[:, 1]).sum()
        total = total + W[t] * (-score + lam * torch.relu(spend - budget) / base)
    return total


def dfl_fit(Xf, Xh, fi, hi, epochs=400, lr=3e-3, lam=4.0, seed=0, taus=(0.6, 0.3, 0.15),
            early=False, patience=40):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    val = rng.random(len(fi)) < (0.2 if early else 0.0)
    trn = ~val
    sc = StandardScaler().fit(Xf[trn])
    A = torch.tensor(sc.transform(Xf[trn]), dtype=torch.float32, device=DEV)
    Bm = torch.tensor(sc.transform(Xh), dtype=torch.float32, device=DEV)

    def pack(mask):
        return (torch.tensor(D[fi[mask]], dtype=torch.float32, device=DEV),
                torch.tensor(np.maximum(DC[fi[mask]], 1e-9), dtype=torch.float32, device=DEV),
                float(TS[fi[mask], 0].sum()), float(TC[fi[mask], 0].sum()), int(mask.sum()))
    dsT, dcT, s0T, baseT, nT = pack(trn)
    if early:
        V = torch.tensor(sc.transform(Xf[val]), dtype=torch.float32, device=DEV)
        dsV, dcV, s0V, baseV, nV = pack(val)
    m = MLP(A.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        tau = taus[min(len(taus) - 1, ep * len(taus) // epochs)]
        total = _surrogate(m(A), dsT, dcT, s0T, baseT, nT, tau, lam)
        opt.zero_grad(set_to_none=True); total.backward(); opt.step(); sch.step()
        if early and ep % 5 == 4:
            with torch.no_grad():
                vl = float(_surrogate(m(V), dsV, dcV, s0V, baseV, nV, taus[-1], lam).item())
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= patience // 5:
                    break
    if early and best_state is not None:
        m.load_state_dict(best_state)
    with torch.no_grad():
        uh = m(Bm).cpu().numpy()
    return np.exp(np.clip(uh, -20, 20))          # slopes


def dfl_head(**kw):
    def f(Xf, Xh, fi, hi):
        slope = dfl_fit(Xf, Xh, fi, hi, **kw)
        # convert the slope back to a gain with the stage's own predicted cost deltas
        leg = Xh[:, 42:45]; lin = Xh[:, 48:51]
        pc = np.exp(0.9 * leg + 0.1 * lin)
        dch = np.column_stack([np.maximum(pc[:, 1] - pc[:, 0], 1e-6),
                               np.maximum(pc[:, 2] - pc[:, 1], 1e-6)])
        return slope * dch
    return f


# ---------------------------------------------------------------- (b) level-2
def level2_head(inner=5, use_ridge=True):
    def f(Xf, Xh, fi, hi):
        rng = np.random.default_rng(5).integers(0, inner, size=len(fi))
        L1f = np.zeros((len(fi), 4))
        for k in range(inner):
            a, b = rng != k, rng == k
            for c in range(2):
                L1f[b, c] = HistGradientBoostingRegressor(**GP).fit(
                    Xf[a], D[fi[a], c]).predict(Xf[b])
                s = StandardScaler().fit(Xf[a])
                L1f[b, 2 + c] = Ridge(alpha=30.0).fit(
                    s.transform(Xf[a]), D[fi[a], c]).predict(s.transform(Xf[b]))
        L1h = np.zeros((len(hi), 4))
        for c in range(2):
            L1h[:, c] = HistGradientBoostingRegressor(**GP).fit(Xf, D[fi, c]).predict(Xh)
            s = StandardScaler().fit(Xf)
            L1h[:, 2 + c] = Ridge(alpha=30.0).fit(s.transform(Xf), D[fi, c]).predict(s.transform(Xh))
        A = np.hstack([Xf, L1f]); Bm = np.hstack([Xh, L1h])
        out = np.zeros((len(hi), 2))
        for c in range(2):
            if use_ridge:
                s = StandardScaler().fit(A)
                out[:, c] = Ridge(alpha=30.0).fit(s.transform(A), D[fi, c]).predict(s.transform(Bm))
            else:
                out[:, c] = HistGradientBoostingRegressor(**GP).fit(A, D[fi, c]).predict(Bm)
        return out
    return f


def assemble(fit_fn):
    t0 = time.perf_counter()
    cvg = np.zeros((len(cv["idx"]), 2))
    for p in parts:
        g = fit_fn(p["Xf"], p["Xh"], p["fit_idx"], p["idx"])
        for j, i in enumerate(p["idx"]):
            cvg[POS[int(i)]] = g[j]
    devg = fit_fn(arr["Xf"], arr["Xh"], arr["fit_idx"], arr["idx"])
    return cvg, devg, time.perf_counter() - t0


if __name__ == "__main__":
    which = sys.argv[1:] or ["dfl", "l2"]
    if "dfl" in which:
        for lab_, kw in (("no early stop, 400 ep", dict(lam=4.0)),
                         ("early stop on the surrogate", dict(lam=4.0, early=True)),
                         ("early stop, 3 seeds", dict(lam=4.0, early=True, seeds=3))):
            seeds = kw.pop("seeds", 1)
            if seeds == 1:
                fn = dfl_head(**kw)
            else:
                def fn(Xf, Xh, fi, hi, kw=kw, seeds=seeds):
                    return np.mean([dfl_head(seed=s, **kw)(Xf, Xh, fi, hi)
                                    for s in range(seeds)], axis=0)
            g, dgv, s = assemble(fn)
            report(f"D1 DFL {lab_}", g, dgv, s)
            report(f"D1 DFL {lab_} ga1 rb0", g, dgv, s, dict(gain_alpha=1.0, rank_beta=0.0))
    if "l2" in which:
        for ur in (True, False):
            g, dgv, s = assemble(level2_head(use_ridge=ur))
            nm = "ridge" if ur else "gbm"
            report(f"D2 level-2 stack ({nm})", g, dgv, s)
            report(f"D2 level-2 ({nm}) ga.85 rb0", g, dgv, s, dict(gain_alpha=0.85, rank_beta=0.0))
