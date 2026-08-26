# SPDX-FileCopyrightText: Copyright 2026 SK TELECOM CO., LTD.
# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
"""b07-visualiser - render every figure from reports/lab/figs/b07_numbers.json.

Palette: Okabe-Ito (colour-blind safe).  Every bar carries its exact number.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports/lab/figs"
NUM = json.loads((FIG / "b07_numbers.json").read_text(encoding="utf-8"))
TIERS = ("fast", "balanced", "premium")
W = {"fast": 0.4, "balanced": 0.3, "premium": 0.3}
MULT = {"fast": 1.25, "balanced": 2.0, "premium": 4.0}
MODELS = ("ax31-light", "ax31 (mid)", "axk1-think")

# Okabe-Ito
BLK, ORA, SKY, GRN, YEL, BLU, VER, PUR = ("#000000", "#E69F00", "#56B4E9", "#009E73",
                                          "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
TIERC = {"fast": SKY, "balanced": ORA, "premium": GRN, "weighted": BLU}
GREY = "#666666"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": ":",
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def bar_labels(ax, rects, vals, fmt="{:.4f}", dy=0.002, fs=6.5, rot=90, color=BLK):
    for r, v in zip(rects, vals):
        ax.text(r.get_x() + r.get_width() / 2, r.get_height() + dy, fmt.format(v),
                ha="center", va="bottom", fontsize=fs, rotation=rot, color=color)


# ============================================================== FIGURE 1
def fig1():
    order = ["all-light", "all-mid", "official hash-regex", "E43 .98/.87/.85",
             "E43 .98/.89/.88", "ceiling (EB latent p)", "oracle (realised s)"]
    nice = {"all-light": "all-light\n(trivial floor)",
            "all-mid": "all-mid\n(busts fast+bal)",
            "official hash-regex": "official\nhash-regex baseline",
            "E43 .98/.87/.85": "deployed E43\n(conservative triple)",
            "E43 .98/.89/.88": "deployed E43\n(published triple)",
            "ceiling (EB latent p)": "noise-free ceiling\n(EB latent p, true cost)",
            "oracle (realised s)": "oracle\n(realised s, true cost)"}
    L = NUM["ladder"]
    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    x = np.arange(len(order))
    wid = 0.2
    for k, t in enumerate(list(TIERS) + ["weighted"]):
        vals, hat, ec = [], [], []
        for c in order:
            if t == "weighted":
                vals.append(L[c]["final"]); hat.append(""); ec.append("none")
            else:
                d = L[c]["tiers"][t]
                vals.append(d["score"])
                hat.append("" if d["passed"] else "////")
                ec.append("none" if d["passed"] else VER)
        rec = ax.bar(x + (k - 1.5) * wid, vals, wid, color=TIERC[t],
                     edgecolor=ec, linewidth=1.2, hatch=None,
                     label=f"{t} (w={W[t]:.1f})" if t != "weighted" else "WEIGHTED FINAL")
        for r, h in zip(rec, hat):
            if h:
                r.set_hatch(h)
        bar_labels(ax, rec, vals, dy=0.004)
        for xi, c in zip(x + (k - 1.5) * wid, order):
            if t != "weighted" and not L[c]["tiers"][t]["passed"]:
                ax.text(xi, 0.30, "BUST", ha="center", va="bottom", fontsize=8,
                        rotation=90, color=VER, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([nice[c] for c in order], fontsize=8.5)
    for xi, c in zip(x, order):
        sf = L[c]["safety"]
        ax.text(xi, -0.115, f"safety {sf}", ha="center", va="top", fontsize=7,
                color=GREY, transform=ax.get_xaxis_transform())
        rr = "  ".join(f"{t[0]}{L[c]['tiers'][t]['ratio']:.2f}" for t in TIERS)
        ax.text(xi, -0.155, f"budget ratio {rr}", ha="center", va="top", fontsize=6.3,
                color=GREY, transform=ax.get_xaxis_transform())
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("mean episode score  (dimensionless, 0-1)")
    ax.set_title("Fig 1 - the ladder: dev-880 final score per tier and weighted\n"
                 "held-out dev, train-only fits; hatched = tier over budget (scores 0 in the final)",
                 fontsize=11, weight="bold")
    ax.legend(ncol=4, loc="upper left", fontsize=8.5, frameon=False)
    ax.axhline(L["E43 .98/.89/.88"]["final"], color=GREY, lw=0.8, ls="--")
    ax.text(-0.45, L["E43 .98/.89/.88"]["final"] + 0.006,
            f"deployed weighted final {L['E43 .98/.89/.88']['final']:.4f}",
            ha="left", fontsize=8, color=GREY)
    ax.text(0.005, -0.235,
            "budget ratio = total cost of the selection / cost if every episode used ax31-light; "
            "caps 1.25 / 2.0 / 4.0.  'weighted' = 0.4*fast + 0.3*balanced + 0.3*premium.",
            transform=ax.transAxes, fontsize=7.2, color=GREY)
    fig.subplots_adjust(bottom=0.19, top=0.88, left=0.05, right=0.99)
    fig.savefig(FIG / "fig1_ladder.png")
    plt.close(fig)


# ============================================================== FIGURE 2
def fig2():
    T = NUM["triples"]
    order = ["EV-optimal", "insurance .93/.80/.75", "deployed .98/.87/.85",
             "baseline-like .96/.91/.88", "published .98/.89/.88"]
    lab = []
    for k in order:
        s = T[k]["safety"]
        lab.append(f"{k.split(' ')[0]}\n{s['fast']:.3f}/{s['balanced']:.3f}/{s['premium']:.3f}")
    ev = [T[k]["EV"] for k in order]
    dv = [T[k]["dev"] for k in order]
    bp = [T[k]["det"]["premium"]["bust"] * 100 for k in order]
    bw = [sum(W[t] * T[k]["det"][t]["bust"] for t in TIERS) * 100 for k in order]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15.5, 6.4),
                                  gridspec_kw=dict(width_ratios=[1.35, 1]))
    x = np.arange(len(order))
    r1 = ax.bar(x - 0.19, ev, 0.36, color=BLU, label="EV  (bootstrap E[final], bust priced in)")
    r2 = ax.bar(x + 0.19, dv, 0.36, color=ORA, label="dev  (single held-out 880 sample)")
    bar_labels(ax, r1, ev, dy=0.002, fs=7.5, rot=0)
    bar_labels(ax, r2, dv, dy=0.002, fs=7.5, rot=0)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8)
    ax.set_ylim(0.55, 0.74)
    ax.set_ylabel("weighted final score (0-1)")
    axb = ax.twinx()
    axb.plot(x, bp, "o-", color=VER, lw=2, ms=7, label="premium bust probability")
    axb.plot(x, bw, "s--", color=PUR, lw=1.6, ms=6, label="weight-averaged bust probability")
    for xi, v in zip(x, bp):
        axb.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=7.5, color=VER, weight="bold")
    for xi, v in zip(x, bw):
        axb.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points", xytext=(0, -13),
                     ha="center", fontsize=7, color=PUR)
    axb.set_ylabel("probability that a tier goes over budget (%)")
    axb.set_ylim(-2, 30)
    axb.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.6, frameon=False, ncol=2)
    ax.set_title("(a) the safety triple trades EV against the dev point estimate",
                 fontsize=10, weight="bold")

    sc = ax2.scatter(ev, dv, c=bp, cmap="plasma_r", s=210, edgecolor=BLK, zorder=3,
                     vmin=0, vmax=25)
    off = {"EV-optimal": (10, -4), "insurance .93/.80/.75": (10, -4),
           "deployed .98/.87/.85": (10, -4), "baseline-like .96/.91/.88": (-8, 12),
           "published .98/.89/.88": (-16, -8)}
    for k, e, d in zip(order, ev, dv):
        ax2.annotate(f"{k}\nEV {e:.4f} / dev {d:.4f}", (e, d), textcoords="offset points",
                     xytext=off[k], fontsize=7.6,
                     ha="left" if off[k][0] > 0 else ("center" if off[k][1] > 0 else "right"))
    cb = fig.colorbar(sc, ax=ax2, pad=0.02)
    cb.set_label("premium bust probability (%)", fontsize=8)
    ax2.set_xlabel("EV - bootstrap expected weighted final (0-1)")
    ax2.set_ylabel("dev - held-out weighted final (0-1)")
    ax2.set_title("(b) EV and dev disagree: the dev-best triple is the EV-worst",
                  fontsize=10, weight="bold")
    ax2.set_xlim(0.598, 0.690); ax2.set_ylim(0.6930, 0.7050)
    fig.suptitle("Fig 2 - EV vs dev for the safety triples that matter "
                 "(bench2: 10-fold OOF over Train for EV, train-only refit for dev)",
                 fontsize=11.5, weight="bold")
    fig.text(0.005, 0.005,
             "EV = mean over 3 seeds x 400 bootstrap resamples of 880 OOF Train rows of "
             "(weighted final, with any over-budget tier scored 0).  Dev is a single sample, "
             "so its ranking is not a decision criterion.",
             fontsize=7.2, color=GREY)
    fig.subplots_adjust(bottom=0.16, top=0.86, left=0.055, right=0.985, wspace=0.28)
    fig.savefig(FIG / "fig2_ev_vs_dev.png")
    plt.close(fig)


# ============================================================== FIGURE 3
def fig3():
    z = np.load(FIG / "b07_curves.npz")
    dc = NUM["dev_curve"]
    T = NUM["triples"]
    marks = {"EV-optimal": (BLU, "-"), "deployed .98/.87/.85": (VER, "--"),
             "published .98/.89/.88": (PUR, ":")}
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.0))
    for ax, t in zip(axes, TIERS):
        g = z[f"{t}_grid"]; ev = z[f"{t}_ev"]; raw = z[f"{t}_raw"]; bu = z[f"{t}_bust"]
        ax.plot(g, ev, color=BLU, lw=2.4, label="EV  (bootstrap, bust priced in)")
        ax.plot(g, raw, color=GRN, lw=1.7, ls="--",
                label="raw score (ignores the bust)")
        ax.plot(dc[t]["grid"], dc[t]["raw"], color=ORA, lw=1.4, ls="-.",
                label="dev raw score (single sample)")
        db = np.array(dc[t]["bust"])
        if db.max() > 0:
            first = float(np.array(dc[t]["grid"])[db > 0].min())
            ax.axvspan(first, g[-1], color=VER, alpha=0.10)
            ax.text(first, 0.845, f" dev busts from s={first:.3f}", fontsize=7,
                    color=VER, va="top")
        ax.set_xlabel("safety ratio applied to the tier budget (dimensionless)")
        ax.set_ylabel("tier mean episode score (0-1)")
        ax.set_ylim(0.0, 0.86)
        axb = ax.twinx()
        axb.plot(g, bu * 100, color=VER, lw=1.8, alpha=0.85,
                 label="bust probability (right axis)")
        axb.fill_between(g, 0, bu * 100, color=VER, alpha=0.10)
        axb.set_ylim(0, 100); axb.grid(False)
        axb.set_ylabel("bust probability (%)", color=VER)
        axb.tick_params(axis="y", colors=VER)
        for j, (k, (c, ls)) in enumerate(marks.items()):
            s = T[k]["safety"][t]
            ax.axvline(s, color=c, ls=ls, lw=1.6)
            i = int(np.argmin(np.abs(g - s)))
            ax.text(0.03, 0.40 - 0.075 * j,
                    f"{k.split(' ')[0]:10s} s={s:.3f}   EV={ev[i]:.4f}   "
                    f"raw={raw[i]:.4f}   bust={bu[i] * 100:.1f}%",
                    transform=ax.transAxes, fontsize=7.4, color=c,
                    family="DejaVu Sans Mono", weight="bold",
                    bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.2))
        ax.set_title(f"{t}  (budget cap {MULT[t]}x all-light, weight {W[t]:.1f})",
                     fontsize=10, weight="bold")
        if t == "fast":
            h1, l1 = ax.get_legend_handles_labels(); h2, l2 = axb.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left", frameon=False)
    fig.suptitle("Fig 3 - per-tier safety curves for the deployed configuration, computed honestly "
                 "(bench2 / protocol.safety_curve; 3 seeds x 400 bootstrap resamples of 880 OOF Train rows)",
                 fontsize=11.5, weight="bold")
    fig.text(0.005, 0.005,
             "The EV curve is the only selection criterion.  Where EV falls away from the raw curve, "
             "the tier is paying the bust cliff.  The dev line is the single held-out sample and is "
             "shown for diagnosis only - it is not what the safety ratio was chosen on.",
             fontsize=7.2, color=GREY)
    fig.subplots_adjust(bottom=0.15, top=0.84, left=0.045, right=0.965, wspace=0.42)
    fig.savefig(FIG / "fig3_safety_curves.png")
    plt.close(fig)


# ============================================================== FIGURE 4
def fig4():
    keys = ["level", "d1", "d2", "cost", "price"]
    nicek = {"level": "level error  (s_light)",
             "d1": "d1 gain error  (s_mid - s_light)",
             "d2": "d2 gain error  (s_k1 - s_mid)",
             "cost": "cost error  (cost inside the utility)",
             "price": "budget limit  (shadow price lambda/L)"}
    col = {"level": GREY, "d1": VER, "d2": ORA, "cost": SKY, "price": PUR}
    groups = list(TIERS) + ["weighted"]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.6))
    for ax, (src, ttl) in zip(axes, ((NUM["decomp"], "(a) gap to the realised-score oracle"),
                                     (NUM["decomp_eb"],
                                      "(b) gap to the noise-free (EB latent-p) oracle"))):
        x = np.arange(len(groups))
        pos = np.zeros(len(groups)); neg = np.zeros(len(groups))
        for k in keys:
            v = np.array([src[g]["phi"][k] for g in groups])
            bot = np.where(v >= 0, pos, neg)
            ax.bar(x, v, 0.55, bottom=bot, color=col[k], edgecolor="white", lw=0.6,
                   label=nicek[k])
            for xi, vi, bi in zip(x, v, bot):
                if abs(vi) > 0.0015:
                    ax.text(xi, bi + vi / 2, f"{vi:+.4f}", ha="center", va="center",
                            fontsize=7.5, color="white" if k != "level" else BLK,
                            weight="bold")
            pos = np.where(v >= 0, pos + v, pos); neg = np.where(v < 0, neg + v, neg)
        tot = np.array([src[g]["gap"] for g in groups])
        ax.plot(x - 0.36, tot, "D", color=BLK, ms=8, zorder=6,
                label="total gap (sum of the five)")
        for xi, v, pv in zip(x, tot, pos):
            ax.annotate(f"total gap {v:+.4f}", (xi, pv + 0.004), ha="center",
                        fontsize=8.5, weight="bold")
            ax.annotate("level = +0.0000 (exactly)", (xi, -0.0265), ha="center",
                        fontsize=6.6, color=GREY)
        ax.axhline(0, color=BLK, lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n{src[g]['deployed']:.4f} -> {src[g]['oracle']:.4f}"
                            for g in groups], fontsize=8.5)
        ax.set_ylabel("share of the dev score gap (episode-score units)")
        ax.set_ylim(-0.032, 0.178)
        ax.set_title(ttl, fontsize=10, weight="bold")
    axes[0].legend(fontsize=7.8, loc="upper left", frameon=False, ncol=2)
    fig.suptitle("Fig 4 - where the gap lives: exact Shapley decomposition of the dev gap "
                 "into level / d1 / d2 / cost / budget-limit",
                 fontsize=11.5, weight="bold")
    fig.text(0.005, 0.012,
             "Five players on the allocator's decision inputs; v(S) = mean realised score of "
             "argmax_m (s_m - price*c_m), price = lambda/L.  The empty coalition reproduces the "
             "deployed selection exactly (asserted in code) and the full\ncoalition reproduces the "
             "oracle selection, so the five values sum to the gap with no residual.  'level' is "
             "exactly 0.0000 because adding a constant to all three of an item's predicted scores "
             "cannot change an argmax -\nthe level channel, which carries most of the project's "
             "headline 'score correlation', is worth nothing to the allocator.  Panel (b) replaces "
             "the realised labels by the empirical-Bayes posterior mean p-hat.",
             fontsize=7.2, color=GREY)
    fig.subplots_adjust(bottom=0.19, top=0.88, left=0.055, right=0.99, wspace=0.22)
    fig.savefig(FIG / "fig4_gap_decomposition.png")
    plt.close(fig)


# ============================================================== FIGURE 5
def fig5():
    PF = NUM["per_family"]
    fams = sorted(PF, key=lambda f: -PF[f]["n"])
    n = len(fams)
    fig = plt.figure(figsize=(16.5, 10.2))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.45, wspace=0.55)
    mc = [SKY, ORA, GRN]

    axa = fig.add_subplot(gs[0, 0:3])
    x = np.arange(n)
    for j in range(3):
        v = [PF[f]["score"][j] for f in fams]
        r = axa.bar(x + (j - 1) * 0.27, v, 0.27, color=mc[j], label=MODELS[j])
        bar_labels(axa, r, v, fmt="{:.3f}", dy=0.006, fs=6.2)
    axa.set_xticks(x); axa.set_xticklabels([f"{f}\nn={PF[f]['n']}" for f in fams], fontsize=7.5)
    axa.set_ylabel("mean realised score on dev (0-1)")
    axa.set_ylim(0, 1.08)
    axa.legend(fontsize=8, ncol=3, frameon=False, loc="upper left")
    axa.set_title("(a) what each model actually scores, by source family (dev 880)",
                  fontsize=10, weight="bold")

    axb = fig.add_subplot(gs[0, 3:6])
    for j in range(3):
        v = [PF[f]["cost_ratio"][j] for f in fams]
        r = axb.bar(x + (j - 1) * 0.27, v, 0.27, color=mc[j], label=MODELS[j])
        for rr, vv in zip(r, v):
            axb.text(rr.get_x() + rr.get_width() / 2, vv * 1.06, f"{vv:.1f}x",
                     ha="center", va="bottom", fontsize=6.2, rotation=90)
    axb.set_yscale("log")
    axb.set_ylim(0.7, 700)
    axb.set_xticks(x); axb.set_xticklabels(fams, fontsize=7.5, rotation=20, ha="right")
    axb.set_ylabel("cost relative to the same family on ax31-light (x, log scale)")
    axb.set_title("(b) what each model costs, by family - the k1 price is family-specific",
                  fontsize=10, weight="bold")
    axb.legend(fontsize=8, ncol=3, frameon=False, loc="upper left")

    short = {"gsm8k_or_other": "gsm8k/other", "truthfulqa": "truthfulqa"}
    for k, t in enumerate(TIERS):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        us = np.array([PF[f][t]["upgrade_share"] for f in fams]) * 100
        gs_ = np.array([PF[f][t]["score_gain_share"] for f in fams]) * 100
        bs = np.array([PF[f][t]["budget_share"] for f in fams]) * 100
        r1 = ax.barh(x + 0.19, us, 0.36, color=VER,
                     label="% of the tier's UPGRADE money (spend above all-light)")
        r2 = ax.barh(x - 0.19, gs_, 0.36, color=BLU,
                     label="% of the tier's SCORE GAIN (score above all-light)")
        for rr, vv in zip(r1, us):
            ax.text(vv + 0.4, rr.get_y() + rr.get_height() / 2, f"{vv:.1f}",
                    va="center", fontsize=6.6, color=VER)
        for rr, vv in zip(r2, gs_):
            ax.text(vv + 0.4, rr.get_y() + rr.get_height() / 2, f"{vv:.1f}",
                    va="center", fontsize=6.6, color=BLU)
        ax.set_yticks(x)
        ax.set_yticklabels([f"{short.get(f, f)}  ({gs_[i] / max(us[i], 1e-9):.2f})\n"
                            f"[{bs[i]:.0f}% of all money]" for i, f in enumerate(fams)],
                           fontsize=6.8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(us.max(), gs_.max()) * 1.30)
        ax.set_xlabel("share of the tier total (%)")
        ax.set_title(f"(c{k+1}) {t}: upgrade money in vs score gain out\n"
                     f"(bracket = gain share / money share; 1.00 = fair)",
                     fontsize=9, weight="bold")
        if k == 0:
            ax.legend(fontsize=7.0, frameon=False, loc="lower right")
    fig.suptitle("Fig 5 - per-family economics of the deployed E43 selection "
                 "(dev 880, safety .98/.89/.88)", fontsize=12, weight="bold")
    fig.text(0.005, 0.005,
             "Panel (b) is why a single global k1 price is wrong: an axk1-think call costs 10.6x a "
             "light call on longdoc and 150.3x on code.\n"
             "Panel (c) measures the decision, not the baseline: 'upgrade money' is the spend "
             "above the all-light cost and 'score gain' is the score above the all-light score, "
             "so a family whose bracket is below 1.00 is buying its upgrades at a worse rate than "
             "the tier average.\n"
             "The second line of each label is that family's share of the tier's TOTAL money, "
             "baseline included - longdoc eats 12-35% of every tier's budget before a single "
             "upgrade is bought, and returns 0.0% of the score gain in all three tiers.",
             fontsize=7.4, color=GREY)
    fig.subplots_adjust(bottom=0.095, top=0.92, left=0.075, right=0.985)
    fig.savefig(FIG / "fig5_per_family.png")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5}[w]()
        print("fig", w, "ok")
