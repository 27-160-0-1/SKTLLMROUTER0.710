# SPDX-License-Identifier: Apache-2.0
"""b04 - can MORE CAPACITY buy the long-k1-output detector a11 asked for?

a11 §4: an oracle veto on items whose k1 output exceeds 5k tokens/generation is
worth +0.0041 EV, and "a detector needs recall, not precision:
value = recall x 0.0041 - FP-cost(veto width)".  a11 could not build the detector.
This is the memory/compute-heavy attempt: train on Train only, score Dev, and
report the achievable recall at each veto width for progressively larger models.
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, importlib
lib = importlib.import_module("b04-memory-heavy_lib")
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

OUT = Path("reports/lab/b04_tailclf.json")
WIDTHS = (0.05, 0.10, 0.15, 0.30)


def report(name, sc_tr, sc_dv, y_dv, res):
    auc = roc_auc_score(y_dv, sc_dv)
    row = {"auc": float(auc)}
    line = f"{name:34s} AUC={auc:.4f} |"
    for w in WIDTHS:
        k = int(round(w * len(sc_dv)))
        top = np.argsort(-sc_dv)[:k]
        rec = float(y_dv[top].sum() / max(y_dv.sum(), 1))
        prec = float(y_dv[top].mean())
        row[f"w{int(w*100)}"] = dict(recall=rec, precision=prec)
        line += f" w{int(w*100)}%: rec {rec:.3f} prec {prec:.3f} |"
    res[name] = row
    print(line, flush=True)


if __name__ == "__main__":
    lab = lib.MemLab(verbose=False)
    tr, dv = lab.train_idx, lab.dev_idx
    opg = lab.otok[:, 2] / np.maximum(lab.ngen[:, 2], 1)
    res = {}
    for thr in (5_000, 10_000):
        y = (opg >= thr).astype(int)
        print(f"\n=== target: k1 output >= {thr} tok/generation "
              f"(train {int(y[tr].sum())}/{len(tr)}, dev {int(y[dv].sum())}/{len(dv)}) ===")
        res[f"thr{thr}"] = r = {}
        # 0. the deployed pipeline's own predicted k1 cost (a11 says this is vacuous)
        Q, V = lib.tfidf_view(lab.C, tr, 256)
        knn = lib.knn_rows(Q, V, tr, dv, lab.targets, 16)
        report("baseline: kNN predicted log c_k1", None, knn[:, 5], y[dv], r)
        # 1. small: 30 dense + 9 family, GBM
        Xd = np.hstack([lab.dense, lab.fam_onehot])
        for it, lv in ((300, 15), (1500, 63)):
            m = HistGradientBoostingClassifier(max_iter=it, learning_rate=0.06, max_leaf_nodes=lv,
                                               min_samples_leaf=20, l2_regularization=3.0,
                                               early_stopping=True, validation_fraction=0.15,
                                               random_state=11).fit(Xd[tr], y[tr])
            report(f"dense30+fam9 GBM ({it} iter, {lv} leaves)", None,
                   m.decision_function(Xd[dv]), y[dv], r)
        # 2. big: + the 16,414 hashed n-gram block via a logistic head, stacked
        lr = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear").fit(lab.X[tr], y[tr])
        s_lr = lr.decision_function(lab.X)
        report("hashed-16k logistic (C=1)", None, s_lr[dv], y[dv], r)
        inner = np.random.default_rng(3).integers(0, 5, size=len(tr))
        oof = np.zeros(len(tr))
        for k in range(5):
            f = tr[inner != k]
            if y[f].sum() < 2:
                continue
            oof[inner == k] = LogisticRegression(C=1.0, max_iter=2000, solver="liblinear").fit(
                lab.X[f], y[f]).decision_function(lab.X[tr[inner == k]])
        Xs_tr = np.hstack([Xd[tr], oof[:, None]])
        Xs_dv = np.hstack([Xd[dv], s_lr[dv][:, None]])
        for it, lv in ((300, 15), (1500, 63)):
            m = HistGradientBoostingClassifier(max_iter=it, learning_rate=0.06, max_leaf_nodes=lv,
                                               min_samples_leaf=20, l2_regularization=3.0,
                                               early_stopping=True, validation_fraction=0.15,
                                               random_state=11).fit(Xs_tr, y[tr])
            report(f"stack dense+fam+hashed ({it} iter, {lv} leaves)", None,
                   m.decision_function(Xs_dv), y[dv], r)
        # 3. regression on log tokens, then threshold (uses the ordering, not the label)
        yl = np.log(np.maximum(opg, 1.0))
        rg = Ridge(alpha=10.0, solver="sparse_cg").fit(lab.X[tr], yl[tr])
        report("hashed-16k ridge on log(out/gen)", None, rg.predict(lab.X[dv]), y[dv], r)
        m = HistGradientBoostingClassifier  # noqa
        from sklearn.ensemble import HistGradientBoostingRegressor
        rr = HistGradientBoostingRegressor(max_iter=1500, learning_rate=0.06, max_leaf_nodes=63,
                                           min_samples_leaf=20, l2_regularization=3.0,
                                           early_stopping=True, validation_fraction=0.15,
                                           random_state=11).fit(
            np.hstack([Xd[tr], rg.predict(lab.X[tr])[:, None]]), yl[tr])
        report("GBM(1500x63) on log(out/gen) + ridge", None,
               rr.predict(np.hstack([Xd[dv], rg.predict(lab.X[dv])[:, None]])), y[dv], r)
    OUT.write_text(json.dumps(res, indent=1, default=float))
