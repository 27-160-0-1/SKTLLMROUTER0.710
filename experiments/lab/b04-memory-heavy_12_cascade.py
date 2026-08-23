# SPDX-License-Identifier: Apache-2.0
"""b04 - cascade / per-family specialisation: is there enough data per family?

Measures (a) how many labelled rows each family actually has, (b) whether a
family-restricted kNN index beats the global one on that family, and (c) the
within-family label variance that a specialised model would have to explain.
No GBM refit - this is the feasibility screen that decides whether the arm is
worth a stage.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, importlib
lib = importlib.import_module("b04-memory-heavy_lib")

OUT = Path("reports/lab/b04_cascade.json")


def cc(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else float("nan")


if __name__ == "__main__":
    lab = lib.MemLab(verbose=False)
    tr, dv = lab.train_idx, lab.dev_idx
    Q, V = lib.tfidf_view(lab.C, tr, 256)
    glob = lib.knn_rows(Q, V, tr, dv, lab.targets, 16)
    ts = lab.true_s[dv]
    td = np.column_stack([ts[:, 1] - ts[:, 0], ts[:, 2] - ts[:, 1]])
    res = {}
    print(f"{'family':16s} {'n_tr':>5} {'n_dv':>5} | {'global d1':>9} {'famonly d1':>10} | "
          f"{'global d2':>9} {'famonly d2':>10} | {'global s_m':>10} {'famonly s_m':>11}")
    for nm in lab.FAMILIES:
        trf = tr[lab.fam_arr[tr] == nm]
        dvf = dv[lab.fam_arr[dv] == nm]
        if len(dvf) < 8 or len(trf) < 8:
            print(f"{nm:16s} {len(trf):5d} {len(dvf):5d} |  (too small)")
            res[nm] = dict(n_tr=len(trf), n_dv=len(dvf))
            continue
        Qf, Vf = lib.tfidf_view(lab.C, trf, 256)
        loc = lib.knn_rows(Qf, Vf, trf, dvf, lab.targets, min(16, len(trf) - 1))
        sel = np.isin(dv, dvf)
        g = glob[sel]
        gd = np.column_stack([g[:, 1] - g[:, 0], g[:, 2] - g[:, 1]])
        ld = np.column_stack([loc[:, 1] - loc[:, 0], loc[:, 2] - loc[:, 1]])
        row = dict(n_tr=int(len(trf)), n_dv=int(len(dvf)),
                   glob_d1=cc(gd[:, 0], td[sel, 0]), loc_d1=cc(ld[:, 0], td[sel, 0]),
                   glob_d2=cc(gd[:, 1], td[sel, 1]), loc_d2=cc(ld[:, 1], td[sel, 1]),
                   glob_sm=cc(g[:, 1], ts[sel, 1]), loc_sm=cc(loc[:, 1], ts[sel, 1]),
                   sd_d1=float(td[sel, 0].std()), sd_d2=float(td[sel, 1].std()))
        res[nm] = row
        print(f"{nm:16s} {row['n_tr']:5d} {row['n_dv']:5d} | {row['glob_d1']:9.3f} {row['loc_d1']:10.3f} | "
              f"{row['glob_d2']:9.3f} {row['loc_d2']:10.3f} | {row['glob_sm']:10.3f} {row['loc_sm']:11.3f}")
    OUT.write_text(json.dumps(res, indent=1, default=float))
