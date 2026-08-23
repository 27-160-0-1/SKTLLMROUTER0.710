# SPDX-License-Identifier: Apache-2.0
"""b07-visualiser figure 6 - project history, claimed gain vs measured gain.

Claimed gains are TRANSCRIBED from EXPERIMENT_LOG.md and each carries the metric
it was claimed on.  Measured values for E43 and E48/C1 were recomputed by
`b07-visualiser_history.py`; for E44/E45/E46 they are read from the run logs
`reports/lab/e4{4,5,6}.log`, and the figure says so.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports/lab/figs"
NUM = json.loads((FIG / "b07_numbers.json").read_text(encoding="utf-8"))
BLK, ORA, SKY, GRN, YEL, BLU, VER, PUR = ("#000000", "#E69F00", "#56B4E9", "#009E73",
                                          "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
GREY = "#666666"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})

# id, verdict, claimed delta, metric of the claim, one-line what
HIST = [
    ("E01", "note", 0.0, "runtime", "pure-python inference 2.0x"),
    ("E02", "note", 0.0, "runtime", "public prompt-hash lookup"),
    ("E03", "adopted", 0.000881, "dev", "linear sweep 1 (4096 bins)"),
    ("E04", "adopted", 0.000170, "dev", "linear sweep 2 (8192 bins)"),
    ("E05", "rejected", 0.0, "dev", "GBM stack with leakage"),
    ("E06", "adopted", 0.001307, "dev", "family mean + kNN"),
    ("E07", "adopted", 0.003000, "EV", "per-tier blend x safety"),
    ("E08", "rejected", 0.0, "EV", "per-item cost cap"),
    ("E09", "note", 0.0, "EV(risk)", "safety on bootstrap EV (+0.154 EV, re-priced risk)"),
    ("E10", "note", 0.0, "-", "private-set risk analysis"),
    ("E11", "adopted", 0.000500, "CV-EV", "train+dev combined fit"),
    ("E12", "note", 0.0, "runtime", "artifact heavy split"),
    ("E13", "rejected", 0.0, "EV", "DeepMind module prior"),
    ("E14", "adopted", 0.003800, "CV", "gain heads"),
    ("E15", "rejected", 0.0, "EV", "word-level kNN"),
    ("E16", "note", 0.0, "-", "Colab MCP infrastructure"),
    ("E17", "rejected", 0.0, "EV", "GBM hyper sweep + seed ensemble"),
    ("E18", "rejected", 0.0, "EV", "MLP meta head"),
    ("E19", "rejected", 0.0, "EV", "char n-gram space sweep"),
    ("E20", "adopted", 0.000800, "EV", "kNN k=16"),
    ("E21", "adopted", 0.001300, "EV", "ordinal score heads"),
    ("E22", "rejected", 0.0, "EV", "feature battery (5 variants)"),
    ("E23b", "rejected", 0.0, "EV", "kNN comp1024"),
    ("E25", "rejected", 0.0, "EV", "sentence-embedding distillation"),
    ("E26", "rejected", 0.0, "EV", "sign-decomposed gain heads"),
    ("E27", "adopted", 0.000700, "EV", "rank-efficiency head"),
    ("E28", "rejected", 0.0, "EV", "supervised selection vocabulary"),
    ("E30", "rejected", 0.0, "EV", "isotonic gain calibration"),
    ("E31", "rejected", 0.0, "EV", "cost-ratio target + smearing"),
    ("E32", "rejected", 0.0, "EV", "per-item cost-uncertainty inflation"),
    ("E34", "rejected", 0.0, "EV", "IPR transplant"),
    ("E35", "rejected", 0.0, "EV", "pure IPR reproduction"),
    ("E36", "rejected", 0.0, "EV", "cost head, dense path"),
    ("E36b", "rejected", 0.0, "EV", "cost head, sparse path"),
    ("E37", "rejected", 0.0, "EV", "MF / conformal cost / exact ILP"),
    ("E38", "rejected", 0.0, "EV", "RouteLLM + LLMRouter, 9 variants"),
    ("E39", "note", 0.0, "EV", "safety-margin stress test"),
    ("E39b", "note", 0.0, "EV", "risk-appetite ladder"),
    ("E40", "rejected", 0.0, "EV", "text augmentation"),
    ("E41", "rejected", 0.0, "CV", "A.X-3.1-Light self-labelling"),
    ("E42", "rejected", 0.0, "EV", "source side-info features"),
    ("E43", "adopted", 0.004000, "CV", "joint hyper-parameter sweep"),
    ("E44", "rejected", 0.0, "cvEV", "per-model/family cost calibration"),
    ("E45", "rejected", 0.0, "cvEV", "gain pair-balance / shrinkage"),
    ("E46", "rejected", 0.0, "cvEV", "coordinate descent on 8 constants"),
    ("E47", "note", 0.0, "-", "CV baseline rebuilt, legacy OOF"),
    ("E48", "candidate", 0.009548, "EV", "legacy head refit out-of-fold (C1)"),
]

# label, claimed, claimed-metric, measured, measured-metric, who measured
CVM = [
    ("E43 joint sweep", 0.0040, "CV EV, 3 seeds (log)", 0.000994,
     "dev, replica @.98/.89/.88", "b07"),
    ("E43 joint sweep (log's own held-out)", 0.0019, "held-out dev (log)", 0.000994,
     "dev, replica @.98/.89/.88", "b07"),
    ("E44 cost-sum calibration", 0.0053, "dev, oracle-tuned (BRIEF)", -0.000538,
     "cvEV, best variant", "e44.log"),
    ("E45 gain shrinkage", 0.0003, "cvEV, best of grid", 0.000028, "dev", "e45.log"),
    ("E46 coordinate descent", 0.0032, "cvEV", -0.268778, "dev (all tiers bust)", "e46.log"),
    ("E48/C1 legacy-OOF meta", 0.0095, "EV (orchestrator)", 0.009548, "EV, bench2", "b07"),
    ("E48/C1 legacy-OOF meta", 0.0023, "dev (orchestrator)", 0.002273, "dev, bench2", "b07"),
]


def fig6():
    H = json.loads((FIG / "b07_history_measured.json").read_text(encoding="utf-8"))
    base = NUM["ladder"]["official hash-regex"]["final"]
    rep = H["replica"]
    e27 = rep["E27 cfg @.98/.89/.88"]["final"]
    e43 = rep["E43 cfg @.98/.89/.88"]["final"]
    e43x = NUM["ladder"]["E43 .98/.89/.88"]["final"]

    fig = plt.figure(figsize=(16.5, 10.0))
    ax = fig.add_axes([0.055, 0.545, 0.925, 0.345])
    ax2 = fig.add_axes([0.315, 0.095, 0.665, 0.305])

    ids = [h[0] for h in HIST]
    x = np.arange(len(HIST))
    cum = base + np.cumsum([h[2] if h[1] == "adopted" else 0.0 for h in HIST])
    ax.step(x, cum, where="post", color=BLU, lw=2.4,
            label="cumulative CLAIMED gain of the adopted experiments (EXPERIMENT_LOG.md)")
    ax.plot([x[-1], x[-1] + 0.9], [cum[-1], cum[-1] + HIST[-1][2]], color=SKY, lw=2.4, ls="--",
            label=f"E48/C1 candidate +{HIST[-1][2]:.4f}, claimed on EV (not deployed)")
    for i, h in enumerate(HIST):
        if h[1] == "adopted" and h[2] > 0:
            ax.annotate(f"{h[0]} +{h[2]:.4f}", (i, cum[i]), textcoords="offset points",
                        xytext=(2, 4 + 14 * (i % 2)), fontsize=7, color=BLU, rotation=30)
        if h[1] == "rejected":
            ax.plot([i], [base - 0.0016], marker="|", color=GREY, ms=8)
    nrej = sum(1 for h in HIST if h[1] == "rejected")
    ax.text(len(HIST) * 0.5, base - 0.0031,
            f"grey ticks = {nrej} rejected experiments of {len(HIST)} logged "
            f"({nrej / len(HIST) * 100:.0f}%)", fontsize=8, color=GREY, ha="center")

    mx = [0, ids.index("E27"), ids.index("E43")]
    my = [base, e27, e43]
    ax.plot(mx, my, "o-", color=VER, lw=2.6, ms=10, zorder=5,
            label="MEASURED held-out dev, train-only fit, safety .98/.89/.88 (recomputed here)")
    ax.plot([ids.index("E43")], [e43x], "s", color=ORA, ms=10, zorder=5,
            label="MEASURED held-out dev of the shipped artifact's own predictions")
    for xi, yi, li, cc in ((mx[0], my[0], f"official hash-regex baseline {base:.6f}", VER),
                           (mx[1], my[1], f"E27 constants (pre-E43) {e27:.6f}", VER),
                           (mx[2], my[2], f"E43 constants, replica {e43:.6f}", VER),
                           (ids.index("E43"), e43x,
                            f"E43, shipped pipeline {e43x:.6f}", ORA)):
        ax.annotate(li, (xi, yi), textcoords="offset points",
                    xytext=(6, 7 if cc == VER else -15), fontsize=8, color=cc)
    ax.annotate("", xy=(len(HIST) - 1.6, cum[-1]), xytext=(len(HIST) - 1.6, e43),
                arrowprops=dict(arrowstyle="<->", color=BLK, lw=1.6))
    ax.text(len(HIST) - 2.2, (cum[-1] + e43) / 2,
            f"claimed  {cum[-1] - base:+.4f}\nmeasured {e43 - base:+.4f}\n"
            f"shortfall {e43 - cum[-1]:+.4f}\n({(e43 - base) / (cum[-1] - base) * 100:.0f}% survived)",
            fontsize=8.5, ha="right", va="center", weight="bold", family="DejaVu Sans Mono",
            bbox=dict(facecolor="white", edgecolor=BLK, alpha=0.92, pad=3))
    ax.set_xticks(x)
    ax.set_xticklabels(ids, fontsize=6.8, rotation=90)
    ax.set_ylabel("weighted final score on dev 880 (0-1)")
    ax.set_ylim(base - 0.005, base + 0.024)
    ax.set_title("(a) claimed cumulative gain vs measured held-out dev, E01 - E48",
                 fontsize=10.5, weight="bold")
    ax.legend(fontsize=8, loc="upper left", frameon=False)

    y = np.arange(len(CVM))
    cl = [c[1] for c in CVM]; me = [c[3] for c in CVM]
    meclip = [max(v, -0.0062) for v in me]
    ax2.barh(y + 0.19, cl, 0.36, color=SKY, label="CLAIMED gain, as logged")
    ax2.barh(y - 0.19, meclip, 0.36, color=VER, label="MEASURED gain, honest protocol")
    for yy, v in zip(y + 0.19, cl):
        ax2.text(v + 0.00015, yy, f"{v:+.4f}", va="center", fontsize=8.2, color=BLU)
    for yy, v, vc in zip(y - 0.19, me, meclip):
        ax2.text(vc + (0.00015 if vc >= 0 else -0.00015), yy,
                 f"{v:+.4f}" + ("  (bar clipped)" if v < -0.0062 else ""),
                 va="center", ha="left" if vc >= 0 else "right", fontsize=8.2, color=VER)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{c[0]}\nclaimed on {c[2]}  |  measured as {c[4]}  [{c[5]}]"
                         for c in CVM], fontsize=7.4)
    ax2.invert_yaxis()
    ax2.axvline(0, color=BLK, lw=1)
    ax2.set_xlim(-0.0080, 0.0125)
    ax2.set_xlabel("change in the weighted final score (episode-score units)")
    ax2.set_title("(b) claimed vs measured, for every experiment where both numbers exist",
                  fontsize=10.5, weight="bold")
    ax2.legend(fontsize=8, loc="lower right", frameon=False)

    fig.suptitle("Fig 6 - project history: what was claimed and what survived measurement",
                 fontsize=12.5, weight="bold")
    fig.text(0.005, 0.006,
             "Claimed values are transcribed from EXPERIMENT_LOG.md and sit on heterogeneous "
             "metrics (in-sample dev, CV, CV EV, cvEV, held-out dev) - that heterogeneity is "
             "itself part of the finding.  The measured column for E43 and E48/C1 was recomputed "
             "here (b07-visualiser_history.py);\nfor E44/E45/E46 it is read from the run logs "
             "reports/lab/e44.log, e45.log, e46.log.  E46's measured value is the dev score of "
             "the cvEV-optimal configuration, which busts all three tiers.  E09 re-priced the "
             "risk criterion rather than the score and is excluded from the cumulative line.",
             fontsize=7.2, color=GREY)
    fig.savefig(FIG / "fig6_history.png")
    plt.close(fig)


if __name__ == "__main__":
    fig6()
    print("fig 6 ok")
