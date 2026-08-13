"""
03_baselines.py
---------------
VADER and BERT baselines used for Table 1, Table 2 and Figure 3.

This script is included for transparency. It cannot be re-run from the contents
of this repository, because it requires the raw review texts, which are not
redistributed here in accordance with TripAdvisor's terms of service.

Input harmonization (Section 3.3 of the paper): the three methods have
different native input limits, so VADER and BERT are re-run on the same
500-character strings the LLM received. The untruncated variants are computed
alongside them as the sensitivity check reported in the note to Table 1.

Usage: place the raw file alongside this script and run
       python code/03_baselines.py
"""

import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RAW = "LLM_FULL_ANALYSIS.csv"

# ---------------------------------------------------------------------------
# This must match exactly what the LLM pipeline received. The LLM was given the
# review body only; setting USE_TITLE to True would break the identical-input
# claim underpinning Table 1.
# ---------------------------------------------------------------------------
TRUNC_CHARS = 500
USE_TITLE = False
TEXT_COL = "text"
TITLE_COL = "title"

BERT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"
LABELS = ["Positive", "Neutral", "Negative"]


def parse_rating(v):
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r"(\d+(?:\.\d+)?)", str(v))
    return float(m.group(1)) if m else np.nan


def stars_to_label(s):
    if pd.isna(s):
        return np.nan
    if s >= 4:
        return "Positive"
    if s == 3:
        return "Neutral"
    return "Negative"


def build_inputs(df):
    if USE_TITLE:
        df["input_full"] = (df[TITLE_COL].fillna("").astype(str) + ". "
                            + df[TEXT_COL].astype(str)).str.strip()
    else:
        df["input_full"] = df[TEXT_COL].astype(str).str.strip()
    df["input_trunc"] = df["input_full"].str[:TRUNC_CHARS]

    L = df["input_full"].str.len()
    print(f"N reviews          : {len(df):,}")
    print(f"Mean / median chars: {L.mean():.1f} / {L.median():.0f} "
          f"(SD {L.std():.1f})")
    print(f"Truncated          : {(L > TRUNC_CHARS).sum():,} "
          f"({100*(L > TRUNC_CHARS).mean():.1f}%)")
    return df


def run_vader(df):
    vader = SentimentIntensityAnalyzer()

    def label(text):
        c = vader.polarity_scores(str(text))["compound"]
        if c >= 0.05:
            return "Positive"
        if c <= -0.05:
            return "Negative"
        return "Neutral"

    df["vader_full"] = df["input_full"].apply(label)
    df["vader_trunc"] = df["input_trunc"].apply(label)
    return df


def run_bert(df, batch_size=64):
    tok = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print("BERT device:", device)

    @torch.no_grad()
    def stars(texts):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i:i + batch_size]]
            enc = tok(batch, truncation=True, max_length=512,
                      padding=True, return_tensors="pt").to(device)
            out.extend((model(**enc).logits.argmax(-1) + 1).cpu().tolist())
        return out

    df["bert_full"] = [stars_to_label(s) for s in stars(df["input_full"].tolist())]
    df["bert_trunc"] = [stars_to_label(s) for s in stars(df["input_trunc"].tolist())]
    return df


def normalise_llm(df):
    """The 16 off-schema 'Mixed' outputs are grouped with Neutral (Figure 3)."""
    valid = set(LABELS)
    raw = df["llm_sentiment"].astype(str).str.strip().str.title()
    n_off = (~raw.isin(valid)).sum()
    print(f"Off-schema LLM sentiment labels: {n_off} "
          f"({100*n_off/len(df):.2f}%)")
    df["llm_pred"] = raw.where(raw.isin(valid), "Neutral")
    return df


def metrics_row(y_true, y_pred, name):
    mP, mR, mF1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    wP, wR, wF1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)
    return {"Method": name, "n": len(y_true),
            "Acc": accuracy_score(y_true, y_pred),
            "BAcc": balanced_accuracy_score(y_true, y_pred),
            "mP": mP, "mR": mR, "mF1": mF1,
            "wP": wP, "wR": wR, "wF1": wF1}


def main():
    df = pd.read_csv(RAW)
    df["stars"] = df["rating"].apply(parse_rating)
    df["gt"] = df["stars"].apply(stars_to_label)
    df = df[df["gt"].notna() & df[TEXT_COL].notna()].reset_index(drop=True)

    df = build_inputs(df)
    df = run_vader(df)
    df = run_bert(df)
    df = normalise_llm(df)

    y = df["gt"].values
    num = ["Acc", "BAcc", "mP", "mR", "mF1", "wP", "wR", "wF1"]

    print("\nTABLE 1 — identical 500-character input")
    t1 = pd.DataFrame([
        metrics_row(y, df["vader_trunc"], "VADER"),
        metrics_row(y, df["bert_trunc"], "BERT"),
        metrics_row(y, df["llm_pred"], "LLM"),
    ])
    t1[num] = t1[num].round(3)
    print(t1.to_string(index=False))

    print("\nSENSITIVITY — untruncated baselines (note to Table 1)")
    t2 = pd.DataFrame([
        metrics_row(y, df["vader_full"], "VADER (untruncated)"),
        metrics_row(y, df["bert_full"], "BERT (untruncated)"),
    ])
    t2[num] = t2[num].round(3)
    print(t2.to_string(index=False))

    print("\nTABLE 2 — per-class, LLM")
    print(classification_report(y, df["llm_pred"], labels=LABELS,
                                digits=3, zero_division=0))

    print("FIGURE 3 — confusion matrices")
    for name, col in [("VADER", "vader_trunc"), ("BERT", "bert_trunc"),
                      ("LLM", "llm_pred")]:
        cm = confusion_matrix(y, df[col], labels=LABELS)
        print(f"\n{name} (rows = true, cols = predicted, order {LABELS}):")
        print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())


if __name__ == "__main__":
    main()
