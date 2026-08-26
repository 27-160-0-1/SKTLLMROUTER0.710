# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""b04 - learning curve of the kNN in INDEX SIZE (rows), not in k.

The kNN table is the component that scales directly with memory, but only if
adding rows still buys accuracy.  This measures the dev-side quality of the kNN
block alone (no GBM refit) as a function of how many labelled train rows are in
the index, on the axis BRIEF2 says matters: the two gains d1 = s_mid - s_light and
d2 = s_k1 - s_mid, plus the level corr for reference.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, importlib
lib = importlib.import_module("b04-memory-heavy_lib")

OUT = Path("reports/lab/b04_indexsize.json")


def corrs(pred, true):
    out = []
    for c in range(pred.shape[1]):
        a, b = pred[:, c], true[:, c]
        out.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 else float("nan"))
    return out


if __name__ == "__main__":
    lab = lib.MemLab(verbose=False)
    dev = lab.dev_idx
    ts = lab.true_s[dev]
    td = np.column_stack([ts[:, 1] - ts[:, 0], ts[:, 2] - ts[:, 1]])
    tc = np.log(lab.true_c[dev])
    rng = np.random.default_rng(5)
    res = {}
    print(f"{'rows':>6} {'k':>4} {'trunc':>6} | {'corr s_l':>8} {'s_m':>7} {'s_k':>7} | "
          f"{'corr d1':>8} {'d2':>7} | {'corr lc_l':>9} {'lc_m':>7} {'lc_k':>7} | {'top1':>6}")
    for frac in (0.0625, 0.125, 0.25, 0.5, 1.0):
        n = max(20, int(round(frac * len(lab.train_idx))))
        reps = 3 if frac < 1.0 else 1
        acc = []
        for r in range(reps):
            sub = np.sort(rng.choice(lab.train_idx, size=n, replace=False)) if frac < 1.0 else lab.train_idx
            Q, V = lib.tfidf_view(lab.C, sub, 256)
            row = lib.knn_rows(Q, V, sub, dev, lab.targets, 16)
            acc.append(row)
        row = np.mean(acc, axis=0) if reps > 1 else acc[0]
        cs = corrs(row[:, :3], ts)
        pd_ = np.column_stack([row[:, 1] - row[:, 0], row[:, 2] - row[:, 1]])
        cd = corrs(pd_, td)
        cc = corrs(row[:, 3:6], tc)
        res[f"n{n}"] = dict(n=n, score=cs, gain=cd, logcost=cc, top1=float(row[:, 6].mean()))
        print(f"{n:6d} {16:4d} {256:6d} | {cs[0]:8.4f} {cs[1]:7.4f} {cs[2]:7.4f} | "
              f"{cd[0]:8.4f} {cd[1]:7.4f} | {cc[0]:9.4f} {cc[1]:7.4f} {cc[2]:7.4f} | {row[:,6].mean():6.3f}")
    print()
    # k and truncation at full index size
    for k in (4, 8, 16, 32, 64, 128, 256):
        for tcp in (256, 0):
            Q, V = lib.tfidf_view(lab.C, lab.train_idx, tcp)
            row = lib.knn_rows(Q, V, lab.train_idx, dev, lab.targets, k)
            cs = corrs(row[:, :3], ts)
            pd_ = np.column_stack([row[:, 1] - row[:, 0], row[:, 2] - row[:, 1]])
            cd = corrs(pd_, td)
            cc = corrs(row[:, 3:6], tc)
            res[f"k{k}_t{tcp}"] = dict(k=k, trunc=tcp, score=cs, gain=cd, logcost=cc)
            print(f"{1760:6d} {k:4d} {tcp if tcp else 0:6d} | {cs[0]:8.4f} {cs[1]:7.4f} {cs[2]:7.4f} | "
                  f"{cd[0]:8.4f} {cd[1]:7.4f} | {cc[0]:9.4f} {cc[1]:7.4f} {cc[2]:7.4f} |")
    # word view for reference
    Qw, Vw = lib.tfidf_view(lab.Wd, lab.train_idx, 256)
    row = lib.knn_rows(Qw, Vw, lab.train_idx, dev, lab.targets, 16)
    cs = corrs(row[:, :3], ts)
    pd_ = np.column_stack([row[:, 1] - row[:, 0], row[:, 2] - row[:, 1]])
    cd = corrs(pd_, td); cc = corrs(row[:, 3:6], tc)
    res["word_k16"] = dict(score=cs, gain=cd, logcost=cc)
    print(f"{'word':>6} {16:4d} {256:6d} | {cs[0]:8.4f} {cs[1]:7.4f} {cs[2]:7.4f} | "
          f"{cd[0]:8.4f} {cd[1]:7.4f} | {cc[0]:9.4f} {cc[1]:7.4f} {cc[2]:7.4f} |")
    OUT.write_text(json.dumps(res, indent=1, default=float))
