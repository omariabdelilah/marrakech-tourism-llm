"""
01_analysis.py
--------------
Reproduces the multi-dimensional analysis reported in the paper from the
released dataset.

Covers: Tables 3, 4, 5, 9, 10, 11 and the chi-square test of Section 4.8.

Input : data/llm_extracted_signals.csv
Output: outputs/table_*.csv  (one file per table)
        console printout of every reported statistic

Run:    python code/01_analysis.py
"""

import ast
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

DATA = "data/llm_extracted_signals.csv"
OUT = "outputs"

ASPECTS = ["safety", "pricing", "service", "cleanliness", "atmosphere"]

# The ten problem categories reported in Table 4. Poor_cleanliness is part of
# the prompt taxonomy but is reported separately (see the note to Table 4).
PROBLEMS_10 = [
    "Aggressive_vendors", "Overcrowding", "Navigation_difficulty", "Scam",
    "Safety", "Other", "High_prices", "Poor_condition", "Getting_lost",
    "Bad_smell",
]

EMOTIONS_10 = [
    "Joy", "Frustration", "Anger", "Fear", "Disgust",
    "Sadness", "Surprise", "Trust", "Neutral", "Other",
]


def load():
    df = pd.read_csv(DATA)
    df["problems"] = df["llm_problems"].apply(parse_list)
    df["keywords"] = df["llm_keywords"].apply(parse_list)
    return df


def parse_list(v):
    """The problems and keywords columns hold JSON-style lists as strings."""
    if pd.isna(v):
        return []
    try:
        out = ast.literal_eval(v)
        return out if isinstance(out, list) else []
    except (ValueError, SyntaxError):
        return []


def save(table, name):
    path = f"{OUT}/{name}.csv"
    table.to_csv(path, index=False)
    print(f"  -> {path}")


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


# --------------------------------------------------------------------------
# Table 3 — Aspect-based sentiment distribution
# --------------------------------------------------------------------------
def table3(df):
    banner("TABLE 3 — Aspect-based sentiment distribution (%)")
    rows = []
    n = len(df)
    for a in ASPECTS:
        vc = df[f"llm_{a}"].value_counts()
        rows.append({
            "Aspect": a.capitalize(),
            "Positive": round(100 * vc.get("Positive", 0) / n, 1),
            "Negative": round(100 * vc.get("Negative", 0) / n, 1),
            "Neutral": round(100 * vc.get("Neutral", 0) / n, 1),
            "N/A": round(100 * vc.get("Not_mentioned", 0) / n, 1),
        })
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))

    # Off-schema labels, reported in the note to Table 3
    print("\nOff-schema aspect labels (note to Table 3):")
    valid = {"Positive", "Negative", "Neutral", "Not_mentioned"}
    for a in ASPECTS:
        off = df.loc[~df[f"llm_{a}"].isin(valid), f"llm_{a}"]
        if len(off):
            share = 100 * len(off) / n
            print(f"  {a:12s} {dict(off.value_counts())}  ({share:.2f}%)")

    save(t, "table3_aspects")
    return t


# --------------------------------------------------------------------------
# Table 4 — Top reported problems
# --------------------------------------------------------------------------
def table4(df):
    banner("TABLE 4 — Top reported problems")
    n = len(df)
    counts = Counter(p for lst in df["problems"] for p in lst)

    rows = [{
        "Problem": p.replace("_", " ").capitalize(),
        "Count": counts.get(p, 0),
        "%": round(100 * counts.get(p, 0) / n, 1),
    } for p in PROBLEMS_10]
    t = pd.DataFrame(rows).sort_values("Count", ascending=False)
    print(t.to_string(index=False))

    pc = counts.get("Poor_cleanliness", 0)
    print(f"\nPoor_cleanliness (reported separately): {pc} ({100*pc/n:.1f}%)")

    taxonomy = set(PROBLEMS_10) | {"Poor_cleanliness"}
    free = {k: v for k, v in counts.items() if k not in taxonomy}
    print(f"Free-form labels outside the taxonomy: {sum(free.values())} mentions, "
          f"{len(free)} distinct strings")
    print(f"  most frequent: {Counter(free).most_common(3)}")

    save(t, "table4_problems")
    return t


# --------------------------------------------------------------------------
# Table 5 — Economic signals
# --------------------------------------------------------------------------
def table5(df):
    banner("TABLE 5 — Economic signals (%)")
    n = len(df)
    schema = {
        "Price perception": ("llm_price_perception",
                             ["Not_mentioned", "Cheap", "Expensive", "Fair"]),
        "Recommendation": ("llm_recommendation",
                           ["Yes", "Not_mentioned", "No"]),
        "Revisit intention": ("llm_revisit",
                              ["Not_mentioned", "No", "Yes"]),
    }
    rows = []
    for signal, (col, cats) in schema.items():
        vc = df[col].value_counts()
        for c in cats:
            rows.append({
                "Signal": signal,
                "Category": c.replace("_", " "),
                "Count": int(vc.get(c, 0)),
                "%": round(100 * vc.get(c, 0) / n, 1),
            })
        off = {k: int(v) for k, v in vc.items() if k not in cats}
        if off:
            rows.append({"Signal": signal, "Category": "(off-schema)",
                         "Count": sum(off.values()),
                         "%": round(100 * sum(off.values()) / n, 2)})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    save(t, "table5_economic_signals")
    return t


# --------------------------------------------------------------------------
# Table 9 — Problem mention rates by site
# --------------------------------------------------------------------------
def table9(df):
    banner("TABLE 9 — Problem mention rates by heritage site (%)")
    rows = []
    for p in PROBLEMS_10:
        rec = {"Problem": p.replace("_", " ").capitalize()}
        rates = {}
        for site in ["medina", "bahia"]:
            sub = df[df["place"] == site]
            hits = sum(p in lst for lst in sub["problems"])
            rates[site] = 100 * hits / len(sub)
        rec["Medina"] = round(rates["medina"], 1)
        rec["Bahia"] = round(rates["bahia"], 2)
        # Ratio computed on unrounded values, as stated in the paper
        rec["Ratio M/B"] = (round(rates["medina"] / rates["bahia"], 1)
                            if rates["bahia"] > 0 else np.inf)
        rows.append(rec)
    t = pd.DataFrame(rows).sort_values("Ratio M/B", ascending=False)
    print(t.to_string(index=False))
    save(t, "table9_problems_by_site")
    return t


# --------------------------------------------------------------------------
# Table 10 — Sentiment distribution by era
# --------------------------------------------------------------------------
def table10(df):
    banner("TABLE 10 — LLM sentiment distribution by era (%)")
    # The 16 off-schema "Mixed" labels are grouped with Neutral, as in Figure 3
    sent = df["llm_sentiment"].where(
        df["llm_sentiment"].isin(["Positive", "Neutral", "Negative"]), "Neutral")
    rows = []
    for e in ["Pre-COVID", "COVID", "Post-COVID"]:
        m = df["era"] == e
        vc = sent[m].value_counts(normalize=True) * 100
        rows.append({
            "Era": e,
            "n": int(m.sum()),
            "Positive": round(vc.get("Positive", 0), 1),
            "Neutral": round(vc.get("Neutral", 0), 1),
            "Negative": round(vc.get("Negative", 0), 1),
        })
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    save(t, "table10_sentiment_by_era")

    print("\nYear-by-year negative share (Figure 7):")
    by_year = pd.crosstab(df["year"], sent, normalize="index") * 100
    print(by_year.round(1).to_string())
    save(by_year.round(2).reset_index(), "figure7_sentiment_by_year")
    return t


# --------------------------------------------------------------------------
# Table 11 + chi-square — problem x emotion
# --------------------------------------------------------------------------
def table11(df):
    banner("TABLE 11 — Conditional emotion probabilities given problems (%)")
    n = len(df)
    marg = df["llm_emotion"].value_counts(normalize=True) * 100

    focus = ["Aggressive_vendors", "Overcrowding", "Scam", "Safety"]
    show = ["Joy", "Frustration", "Anger", "Fear", "Disgust"]

    rows = []
    for emo in show:
        rec = {"Emotion": emo, "Marg.": round(marg.get(emo, 0), 1)}
        for p in focus:
            sub = df[df["problems"].apply(lambda l: p in l)]
            rec[p.replace("_", " ")] = round(
                100 * (sub["llm_emotion"] == emo).mean(), 1)
        rows.append(rec)
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    save(t, "table11_problem_emotion")

    # Full 10x10 matrix used for Figure 8 and the chi-square test
    banner("CHI-SQUARE — problem x emotion (Section 4.8)")
    mat = pd.DataFrame(
        0, index=PROBLEMS_10, columns=EMOTIONS_10, dtype=int)
    for probs, emo in zip(df["problems"], df["llm_emotion"]):
        if emo not in EMOTIONS_10:
            continue
        for p in probs:
            if p in PROBLEMS_10:
                mat.loc[p, emo] += 1

    chi2, p, dof, _ = chi2_contingency(mat)
    N = int(mat.values.sum())
    v = np.sqrt(chi2 / (N * (min(mat.shape) - 1)))
    print(f"  chi2 = {chi2:.2f}   df = {dof}   p = {p:.3g}   N = {N:,}")
    print(f"  Cramer's V = {v:.3f}")

    # Figure 8 reports P(emotion | problem) over ALL reviews carrying the
    # problem, so rows do not sum to 100: the residual is the small share of
    # reviews whose emotion label falls outside the ten-category schema.
    denom = pd.Series(
        {p: sum(p in l for l in df["problems"]) for p in PROBLEMS_10})
    cond = mat.div(denom, axis=0) * 100
    save(cond.round(1).reset_index().rename(columns={"index": "Problem"}),
         "figure8_problem_emotion_heatmap")

    print("\nConditional lifts reported in the text:")
    print(f"  Safety -> Fear            : {marg.get('Fear',0):.1f}% -> "
          f"{cond.loc['Safety','Fear']:.1f}%  "
          f"({cond.loc['Safety','Fear']/marg.get('Fear',1):.1f}x)")
    print(f"  Aggressive vendors -> Frustration : {marg.get('Frustration',0):.1f}% -> "
          f"{cond.loc['Aggressive_vendors','Frustration']:.1f}%  "
          f"({cond.loc['Aggressive_vendors','Frustration']/marg.get('Frustration',1):.1f}x)")
    return t


# --------------------------------------------------------------------------
# Section 4.6.1 / 4.6.3 — cross-site figures quoted in the text
# --------------------------------------------------------------------------
def cross_site(df):
    banner("SECTION 4.6 — cross-site contrasts quoted in the text")
    for site in ["medina", "bahia"]:
        sub = df[df["place"] == site]
        emo = sub["llm_emotion"].value_counts(normalize=True) * 100
        sent = sub["llm_sentiment"].value_counts(normalize=True) * 100
        print(f"\n{site.upper()} (n = {len(sub):,})")
        print(f"  Joy {emo.get('Joy',0):.1f}%   Frustration {emo.get('Frustration',0):.1f}%   "
              f"Anger {emo.get('Anger',0):.1f}%   Fear {emo.get('Fear',0):.2f}%")
        print(f"  Positive {sent.get('Positive',0):.1f}%   "
              f"Negative {sent.get('Negative',0):.1f}%")
        print(f"  Service negative {100*(sub['llm_service']=='Negative').mean():.1f}%   "
              f"Pricing positive {100*(sub['llm_pricing']=='Positive').mean():.1f}%   "
              f"Atmosphere positive {100*(sub['llm_atmosphere']=='Positive').mean():.1f}%")

    print("\nTrip-type segmentation (Section 4.6.3):")
    sub = df[df["trip_type"].notna()]
    print(f"  trip type available for {len(sub):,} of {len(df):,} "
          f"({100*len(sub)/len(df):.1f}%)")
    for tt in ["Couples", "Friends", "Solo"]:
        g = sub[sub["trip_type"] == tt]
        safety = 100 * g["problems"].apply(lambda l: "Safety" in l).mean()
        fear = 100 * (g["llm_emotion"] == "Fear").mean()
        print(f"  {tt:8s} n = {len(g):5,}   Safety {safety:.1f}%   Fear {fear:.1f}%")


# --------------------------------------------------------------------------
# Section 4.7 — pre/post problem shifts
# --------------------------------------------------------------------------
def temporal_problems(df):
    banner("SECTION 4.7 — problem shifts, pre- vs post-pandemic")
    pre = df[df["era"] == "Pre-COVID"]
    post = df[df["era"] == "Post-COVID"]

    for p in ["Overcrowding", "High_prices", "Poor_condition",
              "Aggressive_vendors"]:
        a = 100 * pre["problems"].apply(lambda l: p in l).mean()
        b = 100 * post["problems"].apply(lambda l: p in l).mean()
        print(f"  {p:20s} {a:5.1f}%  ->  {b:5.1f}%")

    print("\n  Corpus composition (explains the aggregate vendor drop):")
    print(f"    Bahia share pre  : {100*(pre['place']=='bahia').mean():.1f}%")
    print(f"    Bahia share post : {100*(post['place']=='bahia').mean():.1f}%")

    print("\n  Within-site checks:")
    for site, p in [("medina", "Aggressive_vendors"), ("bahia", "Poor_condition")]:
        a = 100 * pre[pre["place"] == site]["problems"].apply(lambda l: p in l).mean()
        b = 100 * post[post["place"] == site]["problems"].apply(lambda l: p in l).mean()
        print(f"    {site} / {p:20s} {a:5.1f}%  ->  {b:5.1f}%")


def main():
    df = load()
    print(f"Loaded {len(df):,} reviews from {DATA}")
    print(f"Sites: {dict(df['place'].value_counts())}")
    print(f"Ground truth: "
          f"{dict((df['gt'].value_counts(normalize=True)*100).round(1))}")

    table3(df)
    table4(df)
    table5(df)
    table9(df)
    table10(df)
    table11(df)
    cross_site(df)
    temporal_problems(df)

    banner("DONE — tables written to outputs/")


if __name__ == "__main__":
    main()
