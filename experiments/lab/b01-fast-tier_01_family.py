# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

"""Is a family-level constant policy as good as the item-level model at 1.25x?

Every family posterior is computed HONESTLY: for the CV rows from the other 9
folds only (fold membership reconstructed exactly from bench2's rng), for the dev
rows from all of Train.  Nothing reads dev outcomes.
"""
from __future__ import annotations
import importlib.util, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("b01lib", HERE / "b01-fast-tier_lib.py")
L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
from harness import Lab, DEPLOYED_CFG, DEPLOYED_EXP, MULTS, W, TIERS  # noqa
import bench2 as B
import protocol as P
from ossp_router import similarity  # noqa

# ------------------------------------------------------- repaired classifier
# copied verbatim from experiments/lab/a08_probe13.py (a08's classify_v3)
_RT_Q = re.compile(r"\nQuestion: ")
_RT_FACT = re.compile(r"\b\w+ is (?:not )?\w+\.")
_DM_EXTRA = re.compile(
    r"^(?:Work out|Which is|What is|Add|Subtract|Total of|Product of|Divide|Multiply|"
    r"Calculate|Simplify|Solve|Evaluate|Round|Sort|Put|Let |Suppose|Differentiate|"
    r"Factor|Expand|In base|Convert|How many|What comes next|List the prime|"
    r"Is \d|Find |Give |Print |Sum |Take |\-?\d[\d ,.eE/*+-]* (?:divided by|times|plus|minus))")
_DM_OP = re.compile(r"^-?[\d./]+ (?:divided by|times|plus|minus) -?[\d./]+\.?$")
_LATEX = re.compile(r"\\(?:frac|sqrt|sum|int|cdot|left|right|text|angle|triangle|overline|log|pi|"
                    r"binom|mathbb|dfrac|le|ge|neq|equiv|pmod)")


def classify_v3(text: str) -> str:
    head = text[:600]
    if similarity._CODE.search(head):
        return "code"
    if similarity._HRMCR_AGE.search(head) or similarity._HRMCR_CAL.search(head[:200]):
        return "hrmcr"
    if similarity._TRUTHFULQA.match(text):
        return "truthfulqa"
    if _RT_Q.search(text) and len(_RT_FACT.findall(text)) >= 3:
        return "ruletaker"
    if similarity._RULETAKER.search(head) and " is " in head:
        return "ruletaker"
    if sum("가" <= ch <= "힣" for ch in head) > 40:
        return "belebele"
    if len(text) > 6_000:
        return "longdoc"
    if _LATEX.search(text) and len(text) < 2_000:
        return "aime"
    body = text.strip()
    if len(body) < 400 and (_DM_EXTRA.match(body) or _DM_OP.match(body)):
        return "dmmath"
    if similarity._DMMATH.match(head) and len(text) < 400:
        return "dmmath"
    return "gsm8k_or_other"


def digit_frac(t):
    return sum(c.isdigit() for c in t) / max(len(t), 1)


lab = Lab()
texts = lab.texts
old = lab.fam_arr
new = np.array([classify_v3(t) for t in texts])
# finer: split the two remaining catch-alls once more
fine = new.copy()
df = np.array([digit_frac(t) for t in texts])
fine = np.array([f + (".dig" if (f == "gsm8k_or_other" and d >= 0.08) else "") for f, d in zip(fine, df)])
nch = np.array([len(t) for t in texts])
fine = np.array([f + (".long" if (f == "code" and n >= np.median(nch[new == "code"])) else "")
                 for f, n in zip(fine, nch)])
print(f"[partition] deployed {len(set(old))} buckets, v3 {len(set(new))}, fine {len(set(fine))}, "
      f"moved v3 {(old != new).sum()}/{len(old)}")

cv0, arr0 = B.stage(lab, dict(DEPLOYED_EXP, legacy_oof_meta=True), tag="legoof")

# ------------------------------------------------- honest per-row family means
FOLDS, SEED = 10, 123
fold_of = np.random.default_rng(SEED).integers(0, FOLDS, size=len(lab.train_idx))


def fam_matrix(part):
    """(row-aligned with cv['idx'], row-aligned with arr['idx']) family-mean targets."""
    tg = lab.targets
    tr = lab.train_idx
    # --- CV rows: mean over the other 9 folds
    cvm = np.zeros((len(cv0["idx"]), 6))
    pos = {int(i): k for k, i in enumerate(cv0["idx"])}
    gl_all = {}
    for f in range(FOLDS):
        fit = tr[fold_of != f]
        hold = tr[fold_of == f]
        gm = tg[fit].mean(axis=0)
        means = {}
        for name in np.unique(part[fit]):
            sel = fit[part[fit] == name]
            means[name] = tg[sel].mean(axis=0) if len(sel) >= 8 else gm
        for i in hold:
            cvm[pos[int(i)]] = means.get(part[i], gm)
    # --- dev rows: mean over all of Train
    gm = tg[tr].mean(axis=0)
    means = {}
    for name in np.unique(part[tr]):
        sel = tr[part[tr] == name]
        means[name] = tg[sel].mean(axis=0) if len(sel) >= 8 else gm
    dvm = np.array([means.get(part[i], gm) for i in arr0["idx"]])
    return cvm, dvm


PARTS = {"deployed9": old, "repaired9": new, "fine": fine}
FM = {k: fam_matrix(v) for k, v in PARTS.items()}


def mk_transform(key, mode, w=1.0, tiers=("fast",)):
    cvm, dvm = FM[key]

    def tf(lab_, arr, ps, pc, tier):
        if tier not in tiers:
            return ps, pc
        M = cvm if len(arr["idx"]) == len(cvm) and arr["idx"][0] == cv0["idx"][0] else dvm
        fs = np.clip(M[:, :3], 0.0, 1.0)
        fc = np.exp(M[:, 3:6])
        out_s = (1 - w) * ps + w * fs
        out_c = pc
        if mode == "both":
            out_c = np.exp((1 - w) * np.log(pc) + w * np.log(fc))
            out_c[:, 1] = np.maximum(out_c[:, 1], out_c[:, 0] * (1 + 1e-12))
            out_c[:, 2] = np.maximum(out_c[:, 2], out_c[:, 1] * (1 + 1e-12))
        return out_s, out_c
    return tf


def show(r, tag):
    d = r["dev_tiers"]["fast"]; e = r["det"]["fast"]
    print(f"{tag:44s} sf={r['safety']['fast']:.3f} EVfast={e['ev']:.6f} "
          f"bust={e['bust']*100:4.1f}%  devfast={d['score']:.6f} r={d['ratio']:.4f} "
          f"mgn={d['margin']*100:5.2f}%  wEV={0.4*e['ev']:.5f}", flush=True)


base = B.run(lab, cv0, arr0, DEPLOYED_CFG, label="item-level (legacy-OOF)", verbose=False)
show(base, "item-level baseline")

for key in PARTS:
    for mode in ("score", "both"):
        r = B.run(lab, cv0, arr0, DEPLOYED_CFG, transform=mk_transform(key, mode),
                  label=f"fam {key}/{mode}", verbose=False)
        show(r, f"family-only {key} ({mode}) w=1.0")

print()
for w in (0.25, 0.5, 0.75, 0.9):
    r = B.run(lab, cv0, arr0, DEPLOYED_CFG, transform=mk_transform("repaired9", "score", w),
              label=f"shrink {w}", verbose=False)
    show(r, f"shrink item->repaired9 score w={w}")
