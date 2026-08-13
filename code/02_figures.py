"""
02_figures.py
-------------
Reproduces Figures 4, 5, 6, 7 and 8 from the released dataset.

Input : data/llm_extracted_signals.csv
Output: outputs/fig4_emotions_problems.{png,pdf}
        outputs/fig5_keywords_by_site.{png,pdf}
        outputs/fig6_emotion_by_site.{png,pdf}
        outputs/fig7_sentiment_over_time.{png,pdf}
        outputs/fig8_problem_emotion_heatmap.{png,pdf}

Run:    python code/02_figures.py
"""

import ast
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = "data/llm_extracted_signals.csv"
OUT = "outputs"
DPI = 600

PROBLEMS_10 = [
    "Aggressive_vendors", "Overcrowding", "Navigation_difficulty", "Scam",
    "Safety", "Other", "High_prices", "Poor_condition", "Getting_lost",
    "Bad_smell",
]
EMOTIONS_10 = [
    "Joy", "Frustration", "Anger", "Fear", "Disgust",
    "Sadness", "Surprise", "Trust", "Neutral", "Other",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
})


def parse_list(v):
    if pd.isna(v):
        return []
    try:
        out = ast.literal_eval(v)
        return out if isinstance(out, list) else []
    except (ValueError, SyntaxError):
        return []


def load():
    df = pd.read_csv(DATA)
    df["problems"] = df["llm_problems"].apply(parse_list)
    df["keywords"] = df["llm_keywords"].apply(parse_list)
    return df


def savefig(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"  -> {OUT}/{name}.png / .pdf")


# --------------------------------------------------------------------------
# Figure 4 — emotional tone and top problems
# --------------------------------------------------------------------------
def fig4(df):
    n = len(df)
    emo = df["llm_emotion"].value_counts()
    emo = emo[emo.index.isin(EMOTIONS_10)].head(8)

    counts = Counter(p for lst in df["problems"] for p in lst)
    prob = pd.Series({p: counts.get(p, 0) for p in PROBLEMS_10}
                     ).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    y = np.arange(len(emo))[::-1]
    ax.barh(y, emo.values, color=plt.cm.Set2(np.linspace(0, 1, len(emo))))
    ax.set_yticks(y)
    ax.set_yticklabels(emo.index)
    ax.set_xlabel("Count")
    ax.set_title("(a) Emotional tone", fontweight="bold")
    for yi, v in zip(y, emo.values):
        ax.text(v + n * 0.004, yi, f"{v:,} ({100*v/n:.1f}%)",
                va="center", fontsize=8)
    ax.set_xlim(0, emo.values.max() * 1.28)

    ax = axes[1]
    y = np.arange(len(prob))[::-1]
    ax.barh(y, prob.values, color=plt.cm.tab10(np.linspace(0, 1, len(prob))))
    ax.set_yticks(y)
    ax.set_yticklabels([p.replace("_", " ").capitalize() for p in prob.index])
    ax.set_xlabel("Count")
    ax.set_title("(b) Top problems", fontweight="bold")
    for yi, v in zip(y, prob.values):
        ax.text(v + n * 0.004, yi, f"{v:,} ({100*v/n:.1f}%)",
                va="center", fontsize=8)
    ax.set_xlim(0, prob.values.max() * 1.28)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    savefig(fig, "fig4_emotions_problems")


# --------------------------------------------------------------------------
# Figure 5 — top keywords per site
# --------------------------------------------------------------------------
def fig5(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (site, label, color) in zip(
            axes, [("medina", "Medina", "#c0392b"),
                   ("bahia", "Bahia Palace", "#2e6da4")]):
        sub = df[df["place"] == site]
        kw = Counter(k.strip().lower()
                     for lst in sub["keywords"] for k in lst if k.strip())
        top = pd.Series(dict(kw.most_common(15)))
        y = np.arange(len(top))[::-1]
        ax.barh(y, top.values, color=color, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(top.index)
        ax.set_xlabel("Frequency")
        ax.set_title(f"{label} — Top 15 keywords", fontweight="bold")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig5_keywords_by_site")


# --------------------------------------------------------------------------
# Figure 6 — emotional tone by site
# --------------------------------------------------------------------------
def fig6(df):
    order = ["Joy", "Frustration", "Neutral", "Surprise", "Trust",
             "Anger", "Disgust", "Fear", "Sadness"]
    med = df[df["place"] == "medina"]
    bah = df[df["place"] == "bahia"]
    a = [100 * (med["llm_emotion"] == e).mean() for e in order]
    b = [100 * (bah["llm_emotion"] == e).mean() for e in order]

    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - w/2, a, w, label=f"Medina ($n$={len(med):,})", color="#c0392b")
    ax.bar(x + w/2, b, w, label=f"Bahia Palace ($n$={len(bah):,})",
           color="#2e6da4")
    for xi, (va, vb) in enumerate(zip(a, b)):
        if max(va, vb) > 5:
            ax.text(xi - w/2, va + 0.6, f"{va:.1f}", ha="center", fontsize=8)
            ax.text(xi + w/2, vb + 0.6, f"{vb:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("Share of reviews (%)")
    ax.set_title("Emotional tone distribution: Medina vs. Bahia Palace")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig6_emotion_by_site")


# --------------------------------------------------------------------------
# Figure 7 — sentiment over time
# --------------------------------------------------------------------------
def fig7(df, start_year=2011):
    sent = df["llm_sentiment"].where(
        df["llm_sentiment"].isin(["Positive", "Neutral", "Negative"]),
        "Neutral")
    sub = df[df["year"] >= start_year]
    ct = pd.crosstab(sub["year"], sent[df["year"] >= start_year],
                     normalize="index") * 100

    fig, ax = plt.subplots(figsize=(9, 4.6))
    styles = {"Positive": ("#27ae60", "o"), "Neutral": ("#e8a33d", "s"),
              "Negative": ("#c0392b", "^")}
    for lab, (c, m) in styles.items():
        if lab in ct:
            ax.plot(ct.index, ct[lab], marker=m, color=c, label=lab, lw=1.8,
                    ms=5)

    ax.axvspan(2019.6, 2020.4, color="grey", alpha=0.25,
               label="Pandemic period")
    peak = ct["Negative"].idxmax()
    ax.annotate(f"{ct['Negative'].max():.1f}%",
                xy=(peak, ct["Negative"].max()),
                xytext=(peak - 1.2, ct["Negative"].max() + 5),
                color="#c0392b", fontweight="bold", fontsize=9)

    ax.set_xticks(ct.index)
    ax.set_xticklabels(ct.index, rotation=45)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of reviews (%)")
    ax.set_title(f"Sentiment distribution over time ({start_year}–{ct.index.max()})")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    savefig(fig, "fig7_sentiment_over_time")


# --------------------------------------------------------------------------
# Figure 8 — problem x emotion heatmap
# --------------------------------------------------------------------------
def fig8(df):
    mat = pd.DataFrame(0, index=PROBLEMS_10, columns=EMOTIONS_10, dtype=int)
    for probs, emo in zip(df["problems"], df["llm_emotion"]):
        if emo not in EMOTIONS_10:
            continue
        for p in probs:
            if p in PROBLEMS_10:
                mat.loc[p, emo] += 1

    denom = pd.Series({p: sum(p in l for l in df["problems"])
                       for p in PROBLEMS_10})
    cond = mat.div(denom, axis=0) * 100

    fig, ax = plt.subplots(figsize=(10, 5.4))
    im = ax.imshow(cond.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(EMOTIONS_10)))
    ax.set_xticklabels(EMOTIONS_10, rotation=35, ha="right")
    ax.set_yticks(range(len(PROBLEMS_10)))
    ax.set_yticklabels([p.replace("_", " ").capitalize()
                        for p in PROBLEMS_10])
    ax.set_xlabel("Emotional tone")
    ax.set_ylabel("Reported problem")

    vmax = cond.values.max()
    for i in range(cond.shape[0]):
        for j in range(cond.shape[1]):
            v = cond.iat[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if v > 20 else "normal",
                    color="white" if v > 0.55 * vmax else "#1a1a1a")

    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("P(emotion | problem)  [%]")
    fig.tight_layout()
    savefig(fig, "fig8_problem_emotion_heatmap")


def main():
    df = load()
    print(f"Loaded {len(df):,} reviews")
    fig4(df)
    fig5(df)
    fig6(df)
    fig7(df)
    fig8(df)
    print("\nDone — figures written to outputs/")


if __name__ == "__main__":
    main()
