# %% [markdown]
# # Reproduction notebook
# **A training-free large language model framework for multidimensional analysis of tourism reviews**
#
# This notebook regenerates every table (1–11, A1–A12) and figure (3–8) of the paper from the anonymised files in
# `data/`. It makes **no API calls** and needs no review text.
#
# Main analysis: Gemini 2.5 Flash on the full review text, temperature 0, seed 42,
# thinking budget 8,192, random batches of 15; three independent full-corpus runs (`data/runs/`).
# `data/corpus_llm_outputs.csv` holds run 1 in tabular form. The outputs of the configuration used in the
# preliminary configuration (500 characters, default settings, file-order batches) and the checks run in it are in
# `data/original_run/`.
#
# Outputs (folder `outputs/`):
# * `all_tables.xlsx` – one sheet per table;
# * `fig3_confusion.png` … `fig8_heatmap.png`.
#
# Bootstrap confidence intervals use 2,000 resamples with seed 42; the intervals of the post-stratified precision and recall
# and of the corrected problem rates (Tables 4, 8, 9, A12) use Jeffreys posteriors propagated by simulation (20,000 draws, seed 42). Running the notebook from top to
# bottom reproduces the values reported in the paper. Runtime: about 15 minutes on a laptop.

# %%
import os, json, ast, re, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency, chi2 as chi2_dist, binomtest, norm, pearsonr, spearmanr
from sklearn.metrics import (cohen_kappa_score, accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40)

DATA, OUT = 'data', 'outputs'
os.makedirs(OUT, exist_ok=True)
SEED, B, TRUNC = 42, 2000, 500
TABLES = {}

def RNG(): return np.random.default_rng(SEED)

# %% [markdown]
# ## 1. Load data and preprocessing
# The preprocessing of the paper is reproduced here: star ratings are mapped to three classes
# (4–5 Positive, 3 Neutral, 1–2 Negative); review dates are grouped into periods; review length is the
# length of the whitespace-trimmed text (the only text processing applied before truncation to 500
# characters, which is done in `baselines_vader_bert.ipynb` and `gemini_original_run.ipynb`).
# Off-schema "Mixed" sentiment outputs (3 in run 1) are grouped with Neutral (as in Figure 3).

# %%
PROBLEMS10 = ['Aggressive_vendors', 'Navigation_difficulty', 'Getting_lost', 'Bad_smell', 'Overcrowding',
              'Poor_condition', 'High_prices', 'Scam', 'Safety', 'Other']
PROBLEM_SCHEMA = PROBLEMS10[:7] + ['Poor_cleanliness'] + PROBLEMS10[7:]
EMOTIONS = ['Joy', 'Anger', 'Fear', 'Disgust', 'Sadness', 'Surprise', 'Trust', 'Frustration', 'Neutral', 'Other']
ASPECTS = ['safety', 'pricing', 'service', 'cleanliness', 'atmosphere']
ASPECT_LABELS = ['Positive', 'Negative', 'Neutral', 'Not_mentioned']
LAB = ['Positive', 'Neutral', 'Negative']
PERIOD_BINS = [0, 2019, 2020, 2023, 2025, 9999]
PERIODS = ['2004-2019', '2020', '2021-2023', '2024-2025', '2026 (partial)']

def parse_list(v):
    if isinstance(v, list): return v
    if pd.isna(v): return []
    t = str(v).strip()
    if t in ('', '—', '-', '[]', 'nan', 'None'): return []
    try:
        x = ast.literal_eval(t)
        if isinstance(x, list): return [str(y).strip() for y in x]
    except Exception:
        pass
    return [y.strip() for y in t.split(',') if y.strip()]   # 'A, B' format of the inter-LLM file

def star_label(s): return 'Positive' if s >= 4 else ('Neutral' if s == 3 else 'Negative')

df = pd.read_csv(f'{DATA}/corpus_llm_outputs.csv')
bl = pd.read_csv(f'{DATA}/baselines.csv')
assert (df.row_id == bl.row_id).all()
df = df.merge(bl, on='row_id')
N_ALL = len(df)
MISSING_RUN1 = df[df.llm_sentiment.isna()].row_id.tolist()   # one review without output in run 1
df = df[df.llm_sentiment.notna()].reset_index(drop=True)
df['stars'] = df['rating_raw'].str.extract(r'(\d)')[0].astype(int)
df['star_label'] = df['stars'].apply(star_label)
df['year'] = pd.to_datetime(df['written_date']).dt.year
df['period'] = pd.cut(df['year'], PERIOD_BINS, labels=PERIODS)
df['long'] = df['text_len_chars'] > TRUNC
df['probs'] = df['llm_problems'].apply(parse_list)
raw = df['llm_sentiment'].astype(str).str.strip()
df['llm_pred'] = raw.where(raw.isin(LAB), 'Neutral')
df['llm_neg'] = (df['llm_pred'] == 'Negative').astype(int)
df['star_neg'] = (df['star_label'] == 'Negative').astype(int)
N = len(df)
print(f"Corpus: {N:,} reviews | Medina {(df.site=='medina').sum():,} | Bahia {(df.site=='bahia').sum():,}")
print("Star-derived classes (%):", (df.star_label.value_counts(normalize=True) * 100).round(1).to_dict())
print(f"Reviews > {TRUNC} characters: {df.long.sum():,} | exact duplicate records are kept (no reviews removed)")
print(f"Reviews without LLM output in run 1 (excluded from LLM analyses): {MISSING_RUN1}; analysed: {len(df):,} of {N_ALL:,}")

# %% [markdown]
# ## 2. Common helpers

# %%
def enc(s): return pd.Series(s).map({l: i for i, l in enumerate(LAB)}).values

def fast_metrics(yt, yp, k=3):
    cm = np.bincount(yt * k + yp, minlength=k * k).reshape(k, k)
    tp = np.diag(cm); sup = cm.sum(1); pr = cm.sum(0)
    rec = np.divide(tp, sup, out=np.zeros(k), where=sup > 0)
    prec = np.divide(tp, pr, out=np.zeros(k), where=pr > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    return tp.sum() / cm.sum(), rec[sup > 0].mean(), f1.mean()

def mcnemar(ca, cb):
    b = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb))
    if b + c < 25: return b, c, np.nan, binomtest(b, b + c, 0.5).pvalue
    st = (abs(b - c) - 1) ** 2 / (b + c); return b, c, st, chi2_dist.sf(st, 1)

def compare_methods(y, preds):
    rng = RNG(); yt = enc(y); P = {k: enc(v) for k, v in preds.items()}
    n = len(yt); idx = rng.integers(0, n, (B, n))
    boot = {k: np.array([fast_metrics(yt[i], p[i]) for i in idx]) for k, p in P.items()}
    rows = []
    for k, p in P.items():
        pt = fast_metrics(yt, p); lo, hi = np.percentile(boot[k], [2.5, 97.5], 0)
        rows.append({'Method': k, 'n': n, 'Acc': pt[0], 'Acc_lo': lo[0], 'Acc_hi': hi[0], 'BAcc': pt[1],
                     'BAcc_lo': lo[1], 'BAcc_hi': hi[1], 'mF1': pt[2], 'mF1_lo': lo[2], 'mF1_hi': hi[2]})
    paired = []
    names = list(P)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b_ = names[i], names[j]; d = boot[a] - boot[b_]
            pt = np.array(fast_metrics(yt, P[a])) - np.array(fast_metrics(yt, P[b_]))
            bb, cc, st, pv = mcnemar(P[a] == yt, P[b_] == yt)
            r = {'Comparison': f'{a} - {b_}', 'McNemar_b': bb, 'McNemar_c': cc, 'McNemar_chi2': st, 'McNemar_p': pv}
            for m, nm in enumerate(['Acc', 'BAcc', 'mF1']):
                lo, hi = np.percentile(d[:, m], [2.5, 97.5]); r[f'd{nm}'] = pt[m]; r[f'd{nm}_lo'] = lo; r[f'd{nm}_hi'] = hi
            paired.append(r)
    return pd.DataFrame(rows), pd.DataFrame(paired)

def kappa(a, b, weights=None):
    if weights: return cohen_kappa_score(pd.to_numeric(pd.Series(a)).astype(int), pd.to_numeric(pd.Series(b)).astype(int), weights=weights)
    return cohen_kappa_score(pd.Series(a).astype(str), pd.Series(b).astype(str))

def kappa_ci(a, b, weights=None, rng=None, Bn=B):
    rng = rng or RNG()
    a = np.asarray(pd.Series(a).astype(int if weights else str)); b = np.asarray(pd.Series(b).astype(int if weights else str))
    pt = cohen_kappa_score(a, b, weights=weights); n = len(a); ks = []
    for _ in range(Bn):
        i = rng.integers(0, n, n)
        if len(set(a[i]) | set(b[i])) < 2: continue
        ks.append(cohen_kappa_score(a[i], b[i], weights=weights))
    lo, hi = np.nanpercentile(ks, [2.5, 97.5]); return pt, lo, hi

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z ** 2 / n; c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n * n)) / d; return c - h, c + h

def cramers_v(ct):
    c2, p, dof, _ = chi2_contingency(ct); n = ct.values.sum()
    return c2, p, dof, np.sqrt(c2 / n / max(1, min(ct.shape) - 1))

def tier(k): return 'High' if k >= 0.61 else ('Moderate' if k > 0.40 else 'Low')   # Landis-Koch: substantial / moderate / fair or lower

# %% [markdown]
# ## 3. Table 1 (Panel A), Table 2, Table A1 and Figure 3 — polarity against star-derived labels

# %%
def run_labels(name):
    d = pd.read_csv(f'{DATA}/runs/{name}.csv').drop_duplicates('row_id', keep='last').set_index('row_id')
    js = [json.loads(x) for x in d.raw_json]; js = [j[0] if isinstance(j, list) else j for j in js]
    return pd.Series([str(j.get('overall_sentiment')) for j in js], index=d.index).where(lambda x: x.isin(LAB), 'Neutral')
df['pol_only'] = run_labels('polarity_only').reindex(df.row_id).values
t1a, tA1 = compare_methods(df['star_label'], {'VADER': df['vader_full'], 'BERT': df['bert_full'], 'SiEBERT': df['siebert_full'],
                                              'LLM polarity-only': df['pol_only'], 'LLM': df['llm_pred']})
for m, col in [('VADER', 'vader_full'), ('BERT', 'bert_full'), ('SiEBERT', 'siebert_full'), ('LLM polarity-only', 'pol_only'), ('LLM', 'llm_pred')]:
    for av in ['macro', 'weighted']:
        p, r, f, _ = precision_recall_fscore_support(df.star_label, df[col], labels=LAB, average=av, zero_division=0)
        t1a.loc[t1a.Method == m, f'{av[0]}P'] = p; t1a.loc[t1a.Method == m, f'{av[0]}R'] = r
        t1a.loc[t1a.Method == m, f'{av[0]}F1'] = f
TABLES['Table1_PanelA'] = t1a.round(3); TABLES['TableA1_paired'] = tA1.round(4)
print(t1a.round(3).to_string(index=False)); print(tA1.round(3).to_string(index=False))

# Sensitivity: baselines on the first 500 characters (preliminary configuration)
sens, _ = compare_methods(df['star_label'], {'VADER 500 chars': df['vader_trunc'], 'BERT 500 chars': df['bert_trunc'], 'SiEBERT 500 chars': df['siebert_trunc']})
TABLES['Table1_500char_sensitivity'] = sens.round(3)

# Table 2: per-class, LLM
p, r_, f, s = precision_recall_fscore_support(df.star_label, df.llm_pred, labels=LAB, zero_division=0)
t2 = pd.DataFrame({'Class': LAB, 'Support': s, 'Precision': p, 'Recall': r_, 'F1': f}).round(3)
TABLES['Table2_LLM_per_class'] = t2; print(t2.to_string(index=False))

# Figure 3
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, (m, col) in zip(ax, [('VADER', 'vader_full'), ('BERT', 'bert_full'), ('LLM', 'llm_pred')]):
    cm = confusion_matrix(df.star_label, df[col], labels=LAB)
    a.imshow(cm, cmap='Blues')
    for i in range(3):
        for j in range(3):
            a.text(j, i, f'{cm[i, j]:,}', ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    a.set_xticks(range(3)); a.set_xticklabels(LAB); a.set_yticks(range(3)); a.set_yticklabels(LAB)
    a.set_xlabel('Predicted'); a.set_ylabel('Star-derived label'); a.set_title(m)
    TABLES[f'Fig3_confusion_{m}'] = pd.DataFrame(cm, index=LAB, columns=LAB).reset_index()
plt.tight_layout(); plt.savefig(f'{OUT}/fig3_confusion.png', dpi=200); plt.close()
cmL = confusion_matrix(df.star_label, df.llm_pred, labels=LAB)

# %% [markdown]
# ## 4. Human reference standard: Table 1 (Panel B), Table A2, Tables 7 and 8

# %%
hr = pd.read_csv(f'{DATA}/human_reference_300.csv').set_index('row_id')
S = df.set_index('row_id').loc[hr.index]
HDIMS = ['sentiment', 'score', 'emotion'] + [f'asp_{a}' for a in ASPECTS] + ['price_perception', 'recommendation', 'revisit']
LLMCOL = {'sentiment': 'llm_pred', 'score': 'llm_score', 'emotion': 'llm_emotion', 'price_perception': 'llm_price_perception',
          'recommendation': 'llm_recommendation', 'revisit': 'llm_revisit', **{f'asp_{a}': f'llm_{a}' for a in ASPECTS}}
L = pd.DataFrame({d: S[c].values for d, c in LLMCOL.items()}, index=hr.index)
for p in PROBLEMS10: L[f'p_{p}'] = S['probs'].apply(lambda x, p=p: int(p in x)).values
print('Sample: 100 reviews per star-derived class:', S.star_label.value_counts().to_dict())

PB = {'VADER': S.vader_full.values, 'BERT': S.bert_full.values, 'SiEBERT': S.siebert_full.values, 'LLM polarity-only': S.pol_only.values,
      'LLM': L.sentiment.values, 'LLM single-review calls': run_labels('single_300').reindex(hr.index).values,
      'LLM rubric-aligned': run_labels('rubric_300').reindex(hr.index).values, 'Star-derived labels': S.star_label.values}
t1b, pb = compare_methods(hr['A1_sentiment'], PB)
t1b2, pb2 = compare_methods(hr['A2_sentiment'], PB); TABLES['Table1_PanelB_A2'] = t1b2.round(3); TABLES['Table1_PanelB_A2_paired'] = pb2.round(4)
for m, v in PB.items():
    t1b.loc[t1b.Method == m, 'Acc_vs_A2'] = (np.asarray(v) == hr['A2_sentiment'].values).mean()
cons = hr['A1_sentiment'] == hr['A2_sentiment']
for m, v in PB.items():
    t1b.loc[t1b.Method == m, 'Acc_consensus'] = (np.asarray(v)[cons.values] == hr['A1_sentiment'].values[cons.values]).mean()
TABLES['Table1_PanelB'] = t1b.round(3); TABLES['Table1_PanelB_paired'] = pb.round(4)
print(t1b.round(3).to_string(index=False)); print(pb.round(3).to_string(index=False))

# Table A2
tA2 = (pd.crosstab(S.stars.values, hr.A1_sentiment.values, normalize='index') * 100).round(1)
tA2.insert(0, 'n', pd.Series(S.stars.values).value_counts().sort_index().values)
TABLES['TableA2_human_by_star'] = tA2.reset_index().rename(columns={'row_0': 'stars'})
print(tA2)
k = kappa_ci(S.star_label.values, hr.A1_sentiment.values)

# Tables 7 and 8
rows = []
for d in HDIMS + [f'p_{p}' for p in PROBLEMS10]:
    w = 'quadratic' if d == 'score' else None
    a1, a2, l = hr[f'A1_{d}'], hr[f'A2_{d}'], L[d]
    k1, lo, hi = kappa_ci(a1, l, w); k2, lo2, hi2 = kappa_ci(a2, l, w); hh, hlo, hhi = kappa_ci(a1, a2, w)
    r = {'dimension': d, 'kappa_A1': k1, 'CI_lo': lo, 'CI_hi': hi, 'kappa_A2': k2, 'A2_lo': lo2, 'A2_hi': hi2,
         'kappa_A1_A2': hh, 'HH_lo': hlo, 'HH_hi': hhi, 'kappa_min': min(k1, k2), 'ratio_min_to_HH': min(k1, k2) / hh,
         'accuracy_A1': (a1.astype(str).values == l.astype(str).values).mean(), 'accuracy_A2': (a2.astype(str).values == l.astype(str).values).mean()}
    if w: r['within_1'] = (abs(a1.astype(int).values - l.astype(int).values) <= 1).mean()
    else:
        for av in ['macro', 'weighted']:
            r[f'{av}_F1'] = precision_recall_fscore_support(a1.astype(str), l.astype(str), average=av, zero_division=0)[2]
    if d.startswith('p_'):
        pp, rr, ff, _ = precision_recall_fscore_support(a1.astype(int), l.astype(int), average='binary', zero_division=0)
        r.update({'support': int(a1.sum()), 'P': pp, 'R': rr, 'F1': ff})
    r['tier_conservative'] = tier(min(k1, k2)); r['tier_A1'] = tier(k1); r['tier_A2'] = tier(k2)
    lo_min, hi_min = (lo, hi) if k1 <= k2 else (lo2, hi2); r['CI_crosses_boundary'] = tier(lo_min) != tier(hi_min)
    rows.append(r)
t78 = pd.DataFrame(rows)
TABLES['Table7_8_human_validation'] = t78.round(3); print(t78.round(3).to_string(index=False))

# 3-problem cap: reviews for which annotator 1 listed at most three problems
cap = hr[[f'A1_p_{p}' for p in PROBLEMS10]].sum(1).values <= 3
rows = []
for d in HDIMS + [f'p_{p}' for p in PROBLEMS10]:
    w = 'quadratic' if d == 'score' else None
    rows.append({'dimension': d, 'kappa_all': kappa(hr[f'A1_{d}'], L[d], w), 'kappa_excl_>3_problems': kappa(hr[f'A1_{d}'][cap], L[d][cap], w)})
tsub = pd.DataFrame(rows); TABLES['Sec4_5_subsets'] = tsub.round(3)

# %% [markdown]
# ## 5. Tables 3, 4, 5 and Figures 4, 5 — descriptive outputs

# %%
t3 = pd.DataFrame({a: (df[f'llm_{a}'].value_counts() / N * 100).reindex(ASPECT_LABELS) for a in ASPECTS}).T.round(1)
TABLES['Table3_aspects'] = t3.reset_index(); print(t3)
cnt = pd.Series({p: df.probs.apply(lambda x, p=p: p in x).sum() for p in PROBLEM_SCHEMA}).sort_values(ascending=False)
t4 = pd.DataFrame({'count': cnt, '%': (cnt / N * 100).round(1)})
TABLES['Table4_problems'] = t4.reset_index(); print(t4)
allm = [p for x in df.probs for p in x]; off = [p for p in allm if p not in PROBLEM_SCHEMA]
t5 = pd.concat({s: df[c].value_counts() for s, c in [('Price perception', 'llm_price_perception'),
                                                      ('Recommendation', 'llm_recommendation'), ('Revisit', 'llm_revisit')]})
t5 = pd.DataFrame({'count': t5, '%': (t5 / N * 100).round(1)}); TABLES['Table5_economic'] = t5.reset_index(); print(t5)

emo = df.llm_emotion.value_counts(); top8 = emo[emo.index.isin(EMOTIONS)].head(8)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].barh(top8.index[::-1], (top8 / N * 100)[::-1], color='#4C72B0'); ax[0].set_xlabel('% of reviews'); ax[0].set_title('(a) Emotional tone')
top10 = cnt[cnt.index.isin(PROBLEMS10)].head(10)
ax[1].barh(top10.index[::-1], (top10 / N * 100)[::-1], color='#C44E52'); ax[1].set_xlabel('% of reviews'); ax[1].set_title('(b) Reported problems')
plt.tight_layout(); plt.savefig(f'{OUT}/fig4_emotions_problems.png', dpi=200); plt.close()

kw = df.assign(k=df.llm_keywords.apply(parse_list)).explode('k').dropna(subset=['k'])
kw['k'] = kw['k'].str.lower().str.strip()
t_kw = {s: kw[kw.site == s]['k'].value_counts().head(15) for s in ['medina', 'bahia']}
TABLES['Fig5_keywords'] = pd.DataFrame({f'{s}_keyword': v.index for s, v in t_kw.items()}).join(
    pd.DataFrame({f'{s}_n': v.values for s, v in t_kw.items()}))
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for a, (s, v) in zip(ax, t_kw.items()):
    a.barh(v.index[::-1], v.values[::-1], color='#55A868'); a.set_title('Medina' if s == 'medina' else 'Bahia Palace')
plt.tight_layout(); plt.savefig(f'{OUT}/fig5_keywords.png', dpi=200); plt.close()

# %% [markdown]
# ## 6. Table 6 — inter-LLM consistency (Gemini 2.5 Flash vs Claude Opus 4.7)

# %%
il = pd.read_csv(f'{DATA}/original_run/interllm_gemini_claude_300.csv')   # original configuration
il['star_label'] = il.Stars.apply(star_label)
def gsent(x): return x.astype(str).where(x.astype(str).isin(LAB), 'Neutral')
POL = {'Joy': 'Positive', 'Trust': 'Positive', 'Surprise': 'Positive', 'Neutral': 'Neutral', 'Other': 'Neutral'}
rows = []
def add(name, a, b, w=None):
    k, lo, hi = kappa_ci(a, b, w) if len(set(map(str, a)) | set(map(str, b))) > 1 else (np.nan, np.nan, np.nan)
    rows.append({'dimension': name, 'kappa': k, 'CI_lo': lo, 'CI_hi': hi, 'raw_agreement': (np.asarray(a).astype(str) == np.asarray(b).astype(str)).mean()})
add('Sentiment overall', gsent(il.G_Sent), gsent(il.C_Sent))
for g in LAB: m = il.star_label == g; add(f'Sentiment | star label = {g}', gsent(il.G_Sent[m]), gsent(il.C_Sent[m]))
for s in ['medina', 'bahia']: m = il.Site == s; add(f'Sentiment | {s}', gsent(il.G_Sent[m]), gsent(il.C_Sent[m]))
add('Score (QWK)', il.G_Score, il.C_Score, 'quadratic')
rows.append({'dimension': 'Score within +-1', 'raw_agreement': (abs(il.G_Score - il.C_Score) <= 1).mean()})
add('Emotion (strict)', il.G_Emotion, il.C_Emotion)
add('Emotion (3-class polarity)', il.G_Emotion.map(lambda e: POL.get(e, 'Negative')), il.C_Emotion.map(lambda e: POL.get(e, 'Negative')))
for a, c in [('safety', 'Safety'), ('pricing', 'Pricing'), ('service', 'Service'), ('cleanliness', 'Cleanl.'), ('atmosphere', 'Atmosph.')]:
    add(f'Aspect {a}', il[f'G_{c}'], il[f'C_{c}'])
gp, cp = il.G_Problems.apply(parse_list), il.C_Problems.apply(parse_list)
for p in PROBLEMS10: add(f'Problem {p}', gp.apply(lambda x: int(p in x)), cp.apply(lambda x: int(p in x)))
jac = np.array([1.0 if not (set(a) | set(b)) else len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(gp, cp)])
rows.append({'dimension': 'Problems mean Jaccard', 'raw_agreement': jac.mean()})
for nm, c in [('Price perception', 'Price'), ('Recommendation', 'Recom.'), ('Revisit intention', 'Revisit')]: add(nm, il[f'G_{c}'], il[f'C_{c}'])
t6 = pd.DataFrame(rows); TABLES['Table6_interLLM'] = t6.round(3); print(t6.round(3).to_string(index=False))

# %% [markdown]
# ## 7. Site comparison — Table 9, Figure 6, within-period checks

# %%
t9 = pd.DataFrame({s: [df[df.site == s].probs.apply(lambda x, p=p: p in x).mean() * 100 for p in PROBLEMS10] for s in ['medina', 'bahia']}, index=PROBLEMS10)
t9['ratio'] = t9.medina / t9.bahia; t9 = t9.sort_values('ratio', ascending=False)
TABLES['Table9_site_problems'] = t9.round(2).reset_index(); print(t9.round(2))
es = pd.crosstab(df.llm_emotion, df.site, normalize='columns') * 100
TABLES['Fig6_emotion_by_site'] = es.round(2).reset_index()
e8 = es.loc[top8.index]
e8.plot.bar(figsize=(10, 5), color=['#C44E52', '#4C72B0']); plt.ylabel('% of reviews'); plt.tight_layout()
plt.savefig(f'{OUT}/fig6_emotion_by_site.png', dpi=200); plt.close()
pre = df[df.period == '2004-2019']
wp = {'Joy': [(pre[pre.site == s].llm_emotion == 'Joy').mean() * 100 for s in ['medina', 'bahia']],
      'Negative': [(pre[pre.site == s].llm_pred == 'Negative').mean() * 100 for s in ['medina', 'bahia']],
      'Aggressive vendors': [pre[pre.site == s].probs.apply(lambda x: 'Aggressive_vendors' in x).mean() * 100 for s in ['medina', 'bahia']]}
TABLES['Sec4_6_within_2004_2019'] = pd.DataFrame(wp, index=['medina', 'bahia']).T.round(1).reset_index()
# Trip type and its missingness
tt = df.dropna(subset=['trip_type'])
TABLES['Sec4_6_trip_type_safety'] = pd.DataFrame({g: [tt[tt.trip_type == g].probs.apply(lambda x: 'Safety' in x).mean() * 100]
                                                  for g in ['Couples', 'Friends', 'Solo']}).round(1)
df['tt_missing'] = df.trip_type.isna(); rows = []
for fac in ['site', 'star_label', 'llm_pred', 'long', 'period']:
    c2, p, dof, V = cramers_v(pd.crosstab(df[fac], df.tt_missing))
    rows.append({'factor': fac, 'chi2': c2, 'p': p, 'CramerV': V})
tmiss = pd.DataFrame(rows); TABLES['Sec4_6_triptype_missingness'] = tmiss.round(4)

# %% [markdown]
# ## 8. Temporal analysis — Table 10, Figure 7

# %%
rows = []
for per, g in df.groupby('period', observed=True):
    n = len(g); r = {'period': per, 'n': n}
    for lab, col in [('LLM', 'llm_neg'), ('stars', 'star_neg')]:
        k_ = g[col].sum(); lo, hi = wilson(k_, n); r[f'neg_{lab}_%'] = k_ / n * 100; r[f'{lab}_lo'] = lo * 100; r[f'{lab}_hi'] = hi * 100
    r['mean_LLM_score'] = g.llm_score.mean(); rows.append(r)
t10 = pd.DataFrame(rows); TABLES['Table10_temporal'] = t10.round(2); print(t10.round(1).to_string(index=False))
for per, v in [('2004-2019', 15.2), ('2020', 20.0), ('2021-2023', 20.4), ('2024-2025', 35.2), ('2026 (partial)', 33.3)]:
    pass
c2, p, dof, V = cramers_v(pd.crosstab(df.period, df.llm_neg))
m = smf.logit("llm_neg ~ C(period, Treatment('2004-2019')) + C(site)", data=df).fit(disp=0)
orr = pd.DataFrame({'OR': np.exp(m.params), 'lo': np.exp(m.conf_int()[0]), 'hi': np.exp(m.conf_int()[1])})
m2 = smf.logit("llm_neg ~ C(period, Treatment('2004-2019')) + C(site) + C(trip_type)", data=tt.assign(llm_neg=tt.llm_pred.eq('Negative').astype(int))).fit(disp=0)
orr2 = pd.DataFrame({'OR': np.exp(m2.params), 'lo': np.exp(m2.conf_int()[0]), 'hi': np.exp(m2.conf_int()[1])})
TABLES['Table10_logit'] = orr.round(3).reset_index(); TABLES['Table10_logit_triptype'] = orr2.round(3).reset_index()
yr = df.groupby('year').agg(n=('llm_neg', 'size'), neg=('llm_neg', 'mean'), pos=('llm_pred', lambda x: (x == 'Positive').mean()),
                            neu=('llm_pred', lambda x: (x == 'Neutral').mean()))
TABLES['Fig7_per_year'] = (yr.assign(neg=yr.neg * 100, pos=yr.pos * 100, neu=yr.neu * 100)).round(1).reset_index()
y = yr[yr.index >= 2011]
plt.figure(figsize=(10, 5))
for c, lab, col in [('pos', 'Positive', '#55A868'), ('neu', 'Neutral', '#8C8C8C'), ('neg', 'Negative', '#C44E52')]:
    plt.plot(y.index, y[c] * 100, marker='o', label=lab, color=col)
plt.axvspan(2019.5, 2020.5, color='grey', alpha=.2); plt.ylabel('% of reviews'); plt.legend(); plt.tight_layout()
plt.savefig(f'{OUT}/fig7_temporal.png', dpi=200); plt.close()
ws = {s: [(df[(df.site == s) & (df.period == p)].llm_neg.mean() * 100) for p in ['2004-2019', '2024-2025']] for s in ['bahia', 'medina']}
early, late = df[df.year <= 2019], df[df.year >= 2021]
dec = pd.DataFrame({p: [early.probs.apply(lambda x, p=p: p in x).mean() * 100, late.probs.apply(lambda x, p=p: p in x).mean() * 100]
                    for p in ['Overcrowding', 'High_prices', 'Poor_condition', 'Aggressive_vendors']}, index=['2004-2019', '2021-2026']).T
TABLES['Sec4_7_decomposition'] = dec.round(1).reset_index()

# %% [markdown]
# ## 9. Problem–emotion structure — Table 11, Figure 8, Table A8

# %%
pairs = pd.DataFrame([(p, e) for x, e in zip(df.probs, df.llm_emotion) for p in dict.fromkeys(x) if p in PROBLEMS10 and e in EMOTIONS],
                     columns=['problem', 'emotion'])
ct = pd.crosstab(pairs.problem, pairs.emotion); c2, p, dof, V = cramers_v(ct)
# Table 11 and Figure 8: P(emotion | problem) over all emotion labels (off-schema labels stay in the denominator)
pairs_all = pd.DataFrame([(p, e) for x, e in zip(df.probs, df.llm_emotion) for p in dict.fromkeys(x) if p in PROBLEMS10], columns=['problem', 'emotion'])   # a label repeated within a review counts once
cond = (pd.crosstab(pairs_all.problem, pairs_all.emotion, normalize='index') * 100)[EMOTIONS]
marg = df.llm_emotion.value_counts(normalize=True) * 100
t11 = pd.concat([marg.reindex(['Joy', 'Frustration', 'Anger', 'Fear', 'Disgust']).rename('Marginal'),
                 cond.loc[['Aggressive_vendors', 'Overcrowding', 'Scam', 'Safety'], ['Joy', 'Frustration', 'Anger', 'Fear', 'Disgust']].T], axis=1)
TABLES['Table11_conditional'] = t11.round(1).reset_index(); print(t11.round(1))
plt.figure(figsize=(10, 7)); plt.imshow(cond.values, cmap='Reds')
plt.xticks(range(cond.shape[1]), cond.columns, rotation=45, ha='right'); plt.yticks(range(cond.shape[0]), cond.index)
for i in range(cond.shape[0]):
    for j in range(cond.shape[1]): plt.text(j, i, f'{cond.values[i, j]:.1f}', ha='center', va='center', fontsize=7)
plt.colorbar(label='P(emotion | problem), %'); plt.tight_layout(); plt.savefig(f'{OUT}/fig8_heatmap.png', dpi=200); plt.close()

# Table A8: pairings from human (annotator 1) vs LLM labels on the 300 reviews
def pairsets(src):
    if src == 'human': return [{(p, e) for p in PROBLEMS10 if hr.loc[i, f'A1_p_{p}'] == 1} for i, e in zip(hr.index, hr.A1_emotion)]
    return [{(p, e) for p in PROBLEMS10 if L.loc[i, f'p_{p}'] == 1} for i, e in zip(L.index, L.emotion)]
Hs, Ls = pairsets('human'), pairsets('llm')
idx = [i for i in range(300) if Hs[i] or Ls[i]]
jac8 = np.array([len(Hs[i] & Ls[i]) / len(Hs[i] | Ls[i]) for i in idx]); exact = np.mean([Hs[i] == Ls[i] for i in idx])
rng = RNG(); jb = [jac8[rng.integers(0, len(jac8), len(jac8))].mean() for _ in range(B)]
def condm(sets, emo_series):
    d = pd.DataFrame([pe for s in sets for pe in s], columns=['problem', 'emotion'])
    c = pd.crosstab(d.problem, d.emotion); cp = c.div(c.sum(1), axis=0)
    return cp, cp.div(emo_series.value_counts(normalize=True).reindex(cp.columns), axis=1)
cH, liftH = condm(Hs, hr.A1_emotion); cL, liftL = condm(Ls, L.emotion)
E = sorted(set(cH.columns) | set(cL.columns)); Pp = sorted(set(cH.index) | set(cL.index))
x, y_ = cH.reindex(index=Pp, columns=E, fill_value=0).values.ravel(), cL.reindex(index=Pp, columns=E, fill_value=0).values.ravel()
anyboth = np.zeros(300, bool)
for p in PROBLEMS10: anyboth |= ((hr[f'A1_p_{p}'] == 1).values & (L[f'p_{p}'] == 1).values)
tA8 = pd.DataFrame({'dominant_human': cH.idxmax(1), 'dominant_llm': cL.idxmax(1)}).reindex(Pp)
tA8['same'] = tA8.dominant_human == tA8.dominant_llm
TABLES['TableA8_pairings'] = tA8.reset_index()
TABLES['TableA8_summary'] = pd.DataFrame([{'reviews_with_problem_either': len(idx), 'exact_match': exact, 'mean_jaccard': jac8.mean(),
    'jaccard_lo': np.percentile(jb, 2.5), 'jaccard_hi': np.percentile(jb, 97.5), 'pearson_r_all_cells': pearsonr(x, y_)[0], 'n_cells': len(x),
    'spearman_rho': spearmanr(x, y_)[0], 'same_emotion_when_both_flag': (hr.A1_emotion.values[anyboth] == L.emotion.values[anyboth]).mean()}]).round(3)

# %% [markdown]
# ## 10. Checks in the preliminary configuration (Tables A6, A7), off-schema outputs (Table A3) and cost

# %%
def flat(j):
    d = json.loads(j); o = {}
    if 'overall_sentiment' in d: o['sentiment'] = d['overall_sentiment']
    if 'sentiment_score' in d: o['score'] = d['sentiment_score']
    if 'emotional_tone' in d: o['emotion'] = d['emotional_tone']
    if 'aspects' in d:
        for a in ASPECTS: o[f'asp_{a}'] = (d['aspects'] or {}).get(a)
    if 'problems' in d: o['problems'] = d['problems'] if isinstance(d['problems'], list) else []
    if 'economic_signals' in d:
        e = d['economic_signals'] or {}
        o['price_perception'] = e.get('price_perception'); o['recommendation'] = e.get('recommendation'); o['revisit'] = e.get('revisit_intention')
    return o
def load_run(path):
    d = pd.read_csv(path); return pd.DataFrame([flat(j) for j in d.raw_json], index=d.row_id)
def addp(x):
    x = x.copy()
    for p in PROBLEM_SCHEMA: x[f'p_{p}'] = x['problems'].apply(lambda l, p=p: int(p in (l or [])))
    return x
def kap(a, b, d):
    if d == 'score': return cohen_kappa_score(pd.to_numeric(a).fillna(3).astype(int), pd.to_numeric(b).fillna(3).astype(int), weights='quadratic')
    a, b = a.astype(str), b.astype(str)
    if d == 'sentiment': a, b = a.replace({'Mixed': 'Neutral'}), b.replace({'Mixed': 'Neutral'})
    return cohen_kappa_score(a, b) if len(set(a) | set(b)) > 1 else np.nan
ORIG = pd.read_csv(f'{DATA}/original_run/corpus_llm_outputs_original.csv'); ORIG['probs'] = ORIG.llm_problems.apply(parse_list)
def orig_outputs(ids):
    s = ORIG.set_index('row_id').loc[ids]
    o = pd.DataFrame({'sentiment': s.llm_sentiment.values, 'score': s.llm_score.values, 'emotion': s.llm_emotion.values,
                      'price_perception': s.llm_price_perception.values, 'recommendation': s.llm_recommendation.values,
                      'revisit': s.llm_revisit.values, 'problems': s.probs.values}, index=ids)
    for a in ASPECTS: o[f'asp_{a}'] = s[f'llm_{a}'].values
    return addp(o)
R = f'{DATA}/original_run/reruns'
IDS = list(hr.index)
J = addp(load_run(f'{R}/part0_control.csv').loc[IDS])
C = addp(pd.concat([load_run(f'{R}/partC_{x}.csv') for x in ['sentiment', 'score', 'emotion', 'aspects', 'problems', 'economic']], axis=1).loc[IDS])
O = orig_outputs(IDS)
Hh = pd.DataFrame({d: hr[f'A1_{d}'] for d in HDIMS + [f'p_{p}' for p in PROBLEMS10]})
RDIMS = HDIMS + [f'p_{p}' for p in PROBLEMS10]

# Table A6: joint vs dedicated prompts
rng = RNG(); rows = []
for d in RDIMS:
    h = Hh[d]; kj, ks = kap(J[d], h, d), kap(C[d], h, d); ds = []
    for _ in range(B):
        i = rng.integers(0, 300, 300)
        try: ds.append(kap(C[d].iloc[i], h.iloc[i], d) - kap(J[d].iloc[i], h.iloc[i], d))
        except Exception: pass
    lo, hi = np.nanpercentile(ds, [2.5, 97.5])
    rows.append({'dimension': d, 'joint': kj, 'dedicated': ks, 'diff': ks - kj, 'lo': lo, 'hi': hi, 'significant': lo > 0 or hi < 0,
                 'run_to_run_kappa(original vs same-day joint)': kap(O[d], J[d], d), 'original_vs_human': kap(O[d], h, d)})
tA6 = pd.DataFrame(rows); TABLES['TableA6_joint_vs_dedicated'] = tA6.round(3); print(tA6.round(3).to_string(index=False))
def cramer_pe(x):
    rec = [(p, e) for l, e in zip(x.problems, x.emotion) for p in (l or []) if p in PROBLEMS10]
    c = pd.crosstab(pd.Series([r[0] for r in rec]), pd.Series([r[1] for r in rec])); return cramers_v(c)[3]

# Table A7: cross-language consistency
T = f'{DATA}/original_run/translation'
EN, OR, E2 = [addp(load_run(f'{T}/{x}.csv')).sort_index() for x in ['EN', 'ORIG', 'EN2']]
si = pd.read_csv(f'{T}/sample_index.csv').set_index('row_id').loc[EN.index]
rng = RNG(); rows = []
for d in HDIMS + [f'p_{p}' for p in PROBLEM_SCHEMA]:
    k1, k2 = kap(EN[d], OR[d], d), kap(EN[d], E2[d], d); ks = []
    for _ in range(B):
        i = rng.integers(0, len(EN), len(EN))
        try: ks.append(kap(EN[d].iloc[i], OR[d].iloc[i], d))
        except Exception: pass
    lo, hi = np.nanpercentile(ks, [2.5, 97.5])
    rows.append({'dimension': d, 'EN_vs_ORIG': k1, 'lo': lo, 'hi': hi, 'EN_vs_EN2': k2,
                 'reviews_with_problem': int(((EN[d] + OR[d]) > 0).sum()) if d.startswith('p_') else None})
tA7 = pd.DataFrame(rows); TABLES['TableA7_translation'] = tA7.round(3); print(tA7.round(3).to_string(index=False))
bylang = pd.DataFrame({lang: [kap(EN.loc[gi.index, 'sentiment'], OR.loc[gi.index, 'sentiment'], 'sentiment')] for lang, gi in si.groupby('language')})
TABLES['TableA7_sentiment_by_language'] = bylang.round(2)

# Table A3: off-schema outputs and where they cluster
df['prob_off'] = df.probs.apply(lambda x: any(p not in PROBLEM_SCHEMA for p in x))
SCH = {'llm_sentiment': LAB, 'llm_emotion': EMOTIONS, **{f'llm_{a}': ASPECT_LABELS for a in ASPECTS},
       'llm_price_perception': ['Cheap', 'Fair', 'Expensive', 'Not_mentioned'], 'llm_recommendation': ['Yes', 'No', 'Not_mentioned'],
       'llm_revisit': ['Yes', 'No', 'Not_mentioned']}
df['any_off'] = df.prob_off.copy()
for c, allowed in SCH.items(): df['any_off'] |= ~df[c].astype(str).isin(allowed)
rows = []
for fac, col in [('Site', 'site'), ('Star-derived class', 'star_label'), ('Review length > 500', 'long'), ('Period', 'period')]:
    c2, p, dof, V = cramers_v(pd.crosstab(df[col], df.any_off))
    for lvl, gg in df.groupby(col, observed=True):
        rows.append({'factor': fac, 'level': lvl, 'n': len(gg), 'any_off_%': gg.any_off.mean() * 100, 'problem_off_%': gg.prob_off.mean() * 100, 'p': p, 'CramerV': V})
tA3 = pd.DataFrame(rows); TABLES['TableA3_offschema'] = tA3.round(3); print(tA3.round(2).to_string(index=False))
TABLES['TableA3_offschema_labels'] = pd.Series(off).value_counts().rename_axis('label').reset_index(name='n')

# Cost (Section 3.3)
tok = pd.read_csv(f'{R}/token_log.csv')   # all attempts are billed, including failed ones
def usd(g): return g.prompt_tokens.sum() / 1e6 * 0.30 + (g.output_tokens.sum() + g.thinking_tokens.sum()) / 1e6 * 2.50
summ = pd.read_csv(f'{R}/summary.csv'); cost = pd.DataFrame({'part': summ.part, 'cost_usd': summ.cost_usd})
cost.loc[len(cost)] = ['TOTAL preliminary-configuration checks (token logs)', summ.cost_usd.sum()]
TABLES['Cost_round1_checks'] = cost
s2 = pd.read_csv(f'{DATA}/runs/summary.csv'); tl2 = pd.read_csv(f'{DATA}/runs/token_log.csv')
s2['usd_per_1000_reviews'] = s2.cost_usd / s2.done.str.split('/').str[0].astype(int) * 1000
s2['fallback_requests'] = [int((tl2.part == p + '_fallback').sum()) for p in s2.part]
TABLES['Cost_revision2_runs'] = s2.round(3)

# %% [markdown]
# ## 11. Table A4 — validity evidence behind each management implication

# %%
kk = t78.set_index('dimension')
def fk(d): return f"{kk.loc[d, 'kappa_min']:.3f} ({kk.loc[d, 'tier_conservative']})"
def status(ds): return 'Supported' if all(kk.loc[d, 'tier_conservative'] == 'High' for d in ds) else 'Indicative'
TABLES['TableA4_implications'] = pd.DataFrame([
    ['R1 Commercial actors (Medina)', 'Aggressive vendors', fk('p_Aggressive_vendors'), status(['p_Aggressive_vendors'])],
    ['R1 (emotional co-occurrence)', 'Emotional tone', fk('emotion'), status(['emotion'])],
    ['R2 Restoration and interpretation (Bahia)', 'Poor condition', fk('p_Poor_condition'), status(['p_Poor_condition'])],
    ['R3 2024-2025 deterioration', 'Sentiment', fk('sentiment'), status(['sentiment'])],
    ['R3 (candidate drivers)', 'Overcrowding; high prices; poor condition', '; '.join(fk(d) for d in ['p_Overcrowding', 'p_High_prices', 'p_Poor_condition']),
     status(['p_Overcrowding', 'p_High_prices', 'p_Poor_condition'])],
    ['R4 Safety (Medina)', 'Safety problem', fk('p_Safety'), status(['p_Safety'])]],
    columns=['Implication', 'Dimensions used', 'conservative kappa (tier)', 'Status'])

# %% [markdown]
# ## 11b. Robustness analyses: run-to-run variability, batch composition, rubric alignment,
# ## post-stratification, corrected prevalence, per-emotion metrics and the permutation test

# %%
RUNS = {k: addp(load_run(f'{DATA}/runs/{k}.csv')) for k in ['run1', 'run2', 'run3']}
for k in RUNS: RUNS[k]['sentiment'] = RUNS[k].sentiment.astype(str).where(RUNS[k].sentiment.astype(str).isin(LAB), 'Neutral')
ALLD = HDIMS + [f'p_{p}' for p in PROBLEMS10]
META = df.set_index('row_id')
COMMON = sorted(set(META.index).intersection(*[set(v.index) for v in RUNS.values()]))

# (a) headline metrics per run and their run-to-run SD
rows = []
for k, X in RUNS.items():
    y = enc(META.loc[COMMON, 'star_label']); p = enc(X.loc[COMMON, 'sentiment'])
    acc, bacc, mf1 = fast_metrics(y, p)
    neg = (X.loc[COMMON, 'sentiment'] == 'Negative').groupby(META.loc[COMMON, 'period'].values, observed=True).mean() * 100
    r = {'run': k, 'Acc_vs_stars': acc, 'BAcc_vs_stars': bacc, 'mF1_vs_stars': mf1}
    for d_ in ['sentiment', 'score'] + [f'p_{p}' for p in ['Aggressive_vendors', 'Scam', 'Safety', 'Poor_condition', 'Overcrowding', 'High_prices']]:
        r[f'kappa_A1_{d_}'] = kappa(hr[f'A1_{d_}'], X.loc[hr.index, d_], 'quadratic' if d_ == 'score' else None)
    r['Acc_vs_A1'] = (X.loc[hr.index, 'sentiment'].values == hr.A1_sentiment.values).mean()
    for per in PERIODS: r[f'Neg%_{per}'] = neg.get(per, np.nan)
    for p_ in PROBLEM_SCHEMA: r[f'prob%_{p_}'] = X.loc[COMMON, 'problems'].apply(lambda l, p_=p_: p_ in (l or [])).mean() * 100
    for l_ in LAB: r[f'sent%_{l_}'] = (X.loc[COMMON, 'sentiment'] == l_).mean() * 100
    for s_ in ['medina', 'bahia']:
        m_ = META.loc[COMMON, 'site'].values == s_
        r[f'{s_}_Neg%'] = (X.loc[COMMON, 'sentiment'].values[m_] == 'Negative').mean() * 100
        r[f'{s_}_Joy%'] = (X.loc[COMMON, 'emotion'].values[m_] == 'Joy').mean() * 100
        r[f'{s_}_AV%'] = X.loc[COMMON, 'p_Aggressive_vendors'].values[m_].mean() * 100
    rows.append(r)
rr = pd.DataFrame(rows).set_index('run').T; rr['mean'] = rr.mean(1); rr['sd'] = rr[['run1', 'run2', 'run3']].std(1, ddof=1)
TABLES['R2_run_to_run'] = rr.round(4).reset_index()
rows = []
for a_, b_ in [('run1', 'run2'), ('run1', 'run3'), ('run2', 'run3')]:
    for d_ in ALLD: rows.append({'pair': f'{a_} vs {b_}', 'dimension': d_, 'kappa': kappa(RUNS[a_].loc[COMMON, d_], RUNS[b_].loc[COMMON, d_], 'quadratic' if d_ == 'score' else None)})
TABLES['R2_run_agreement'] = pd.DataFrame(rows).pivot(index='dimension', columns='pair', values='kappa').round(3).reset_index()
# tier stability across runs (conservative kappa)
rows = []
for k, X in RUNS.items():
    for d_ in ALLD:
        w = 'quadratic' if d_ == 'score' else None
        k1, k2 = kappa(hr[f'A1_{d_}'], X.loc[hr.index, d_], w), kappa(hr[f'A2_{d_}'], X.loc[hr.index, d_], w)
        rows.append({'run': k, 'dimension': d_, 'kappa_min': min(k1, k2), 'tier': tier(min(k1, k2))})
_t = pd.DataFrame(rows).pivot(index='dimension', columns='run', values=['kappa_min', 'tier']); _t.columns = [f'{a}_{b}' for a, b in _t.columns]
TABLES['R2_tiers_by_run'] = _t.reset_index()

# (b) batch composition: one review per request (2,000 random reviews; the 300 reference reviews)
SC = addp(load_run(f'{DATA}/runs/single_call.csv')); SC['sentiment'] = SC.sentiment.astype(str).where(SC.sentiment.astype(str).isin(LAB), 'Neutral')
S3 = addp(load_run(f'{DATA}/runs/single_300.csv')); S3['sentiment'] = S3.sentiment.astype(str).where(S3.sentiment.astype(str).isin(LAB), 'Neutral')
RB = addp(load_run(f'{DATA}/runs/rubric_300.csv')); RB['sentiment'] = RB.sentiment.astype(str).where(RB.sentiment.astype(str).isin(LAB), 'Neutral')
ids2 = [i for i in SC.index if i in set(COMMON)]
rows = []
for d_ in ALLD:
    r = {'dimension': d_}
    for k in RUNS: r[f'single_vs_{k}'] = kappa(SC.loc[ids2, d_], RUNS[k].loc[ids2, d_], 'quadratic' if d_ == 'score' else None)
    r['run1_vs_run2'] = kappa(RUNS['run1'].loc[ids2, d_], RUNS['run2'].loc[ids2, d_], 'quadratic' if d_ == 'score' else None)
    rows.append(r)
TABLES['R2_single_vs_batched_kappa'] = pd.DataFrame(rows).round(3)
M2 = META.loc[ids2]; rows = []
for nm, X in [('single-review calls', SC.loc[ids2])] + [(f'{k} (random batches)', v.loc[ids2]) for k, v in RUNS.items()] + [('original run (file-order batches, 500 chars)', orig_outputs(ids2))]:
    r = {'source': nm, 'n': len(ids2)}
    xs = X.sentiment.astype(str).where(X.sentiment.astype(str).isin(LAB), 'Neutral')
    for l_ in LAB: r[f'{l_}%'] = (xs == l_).mean() * 100
    for s_ in ['medina', 'bahia']:
        m_ = (M2.site == s_).values
        r[f'Neg%_{s_}'] = (xs.values[m_] == 'Negative').mean() * 100
        for p_ in ['Aggressive_vendors', 'Scam', 'Safety', 'Overcrowding', 'Poor_condition']: r[f'{p_}%_{s_}'] = X[f'p_{p_}'].values[m_].mean() * 100
    for per in ['2004-2019', '2024-2025']: r[f'Neg%_{per}'] = (xs.values[(M2.period == per).values] == 'Negative').mean() * 100
    r['Acc_vs_stars'] = (xs.values == M2.star_label.values).mean()
    rows.append(r)
TABLES['R2_batch_effect_contrasts'] = pd.DataFrame(rows).round(2)
def human_kappas(X, label):
    rows = []
    for d_ in ALLD:
        w = 'quadratic' if d_ == 'score' else None
        k1, k2 = kappa(hr[f'A1_{d_}'], X.loc[hr.index, d_], w), kappa(hr[f'A2_{d_}'], X.loc[hr.index, d_], w)
        rows.append({'source': label, 'dimension': d_, 'kappa_A1': k1, 'kappa_A2': k2, 'kappa_min': min(k1, k2), 'tier': tier(min(k1, k2))})
    return pd.DataFrame(rows)
hk = pd.concat([human_kappas(RUNS['run1'], 'batched (run 1)'), human_kappas(S3, 'single-review calls'), human_kappas(RB, 'rubric-aligned prompt'),
                human_kappas(orig_outputs(list(hr.index)), 'original configuration')])
TABLES['R2_human_kappa_by_configuration'] = hk.round(3)

# (c) post-stratified agreement (weights = corpus star-class shares) and corrected problem prevalence
Wc = META.star_label.value_counts(normalize=True).reindex(LAB); strata = META.loc[hr.index, 'star_label'].values
def wts(st):
    c_ = pd.Series(st).value_counts(); return np.array([Wc[x] / c_[x] for x in st])
def wmetrics(y, p, w):
    cm = np.zeros((3, 3)); np.add.at(cm, (enc(y), enc(p)), w); tp = np.diag(cm); sup = cm.sum(1); pr = cm.sum(0)
    rec = np.divide(tp, sup, out=np.zeros(3), where=sup > 0); prec = np.divide(tp, pr, out=np.zeros(3), where=pr > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(3), where=(prec + rec) > 0); return np.array([tp.sum() / cm.sum(), rec.mean(), f1.mean()])
rng = RNG(); grp = [np.where(strata == s_)[0] for s_ in LAB]
BI = [np.concatenate([rng.choice(g, len(g)) for g in grp]) for _ in range(B)]
rows = []
for ref in ['A1', 'A2']:
    y = hr[f'{ref}_sentiment'].values
    for m_, v in PB.items():
        v = np.asarray(v); pt = wmetrics(y, v, wts(strata)); bs = np.array([wmetrics(y[i], v[i], wts(strata[i])) for i in BI])
        lo, hi = np.percentile(bs, [2.5, 97.5], 0)
        rows.append({'reference': ref, 'Method': m_, 'Acc_w': pt[0], 'Acc_lo': lo[0], 'Acc_hi': hi[0], 'BAcc_w': pt[1], 'BAcc_lo': lo[1], 'BAcc_hi': hi[1], 'mF1_w': pt[2]})
TABLES['R2_PanelB_poststratified'] = pd.DataFrame(rows).round(3)
def wPR(y, p, w):
    tp = (w * (y == 1) * (p == 1)).sum(); fp = (w * (y == 0) * (p == 1)).sum(); fn = (w * (y == 1) * (p == 0)).sum()
    return (tp / (tp + fp) if tp + fp > 0 else np.nan, tp / (tp + fn) if tp + fn > 0 else np.nan)
cst = META.loc[COMMON, 'star_label'].values
def corr_prev(y, p, st, cp):
    pp = y[p == 1].mean() if (p == 1).any() else 0.0; pf = y[p == 0].mean(); tot = 0
    for s_ in LAB:
        m_ = st == s_; q = cp[cst == s_].mean()
        ppv = y[m_ & (p == 1)].mean() if (m_ & (p == 1)).any() else pp; fo = y[m_ & (p == 0)].mean() if (m_ & (p == 0)).any() else pf
        tot += Wc[s_] * (q * ppv + (1 - q) * fo)
    return tot
rows = []
for p_ in PROBLEMS10:
    pm = L[f'p_{p_}'].values.astype(int); cp = META.loc[COMMON, 'probs'].apply(lambda l, p_=p_: int(p_ in l)).values
    r = {'problem': p_, 'LLM_raw_%': cp.mean() * 100}
    for ref in ['A1', 'A2']:
        y = hr[f'{ref}_p_{p_}'].values.astype(int); Pw, Rw = wPR(y, pm, wts(strata))
        bs = np.array([wPR(y[i], pm[i], wts(strata[i])) for i in BI], dtype=float)
        c_ = corr_prev(y, pm, strata, cp); cb = np.array([corr_prev(y[i], pm[i], strata[i], cp) for i in BI])
        r.update({f'P_{ref}': precision_recall_fscore_support(y, pm, average='binary', zero_division=0)[0],
                  f'R_{ref}': precision_recall_fscore_support(y, pm, average='binary', zero_division=0)[1],
                  f'Pw_{ref}': Pw, f'Pw_lo_{ref}': np.nanpercentile(bs[:, 0], 2.5), f'Pw_hi_{ref}': np.nanpercentile(bs[:, 0], 97.5),
                  f'Rw_{ref}': Rw, f'Rw_lo_{ref}': np.nanpercentile(bs[:, 1], 2.5), f'Rw_hi_{ref}': np.nanpercentile(bs[:, 1], 97.5),
                  f'corrected_%_{ref}': c_ * 100, f'corr_lo_{ref}': np.percentile(cb, 2.5) * 100, f'corr_hi_{ref}': np.percentile(cb, 97.5) * 100})
    r['bound_low_%'] = min(r['corr_lo_A1'], r['corr_lo_A2']); r['bound_high_%'] = max(r['corr_hi_A1'], r['corr_hi_A2']); rows.append(r)
TABLES['R2_corrected_prevalence'] = pd.DataFrame(rows).round(3)

# corrected prevalence by site: same class-specific PPV and false-omission rates, applied to the
# star-class mix and detection rates of each site
csite = META.loc[COMMON, 'site'].values
def corr_prev_site(y, p, st, cp, site):
    pp = y[p == 1].mean() if (p == 1).any() else 0.0; pf = y[p == 0].mean(); tot = 0
    sm = csite == site; ws = pd.Series(cst[sm]).value_counts(normalize=True)
    for s_ in LAB:
        m_ = st == s_; q = cp[sm & (cst == s_)].mean()
        ppv = y[m_ & (p == 1)].mean() if (m_ & (p == 1)).any() else pp; fo = y[m_ & (p == 0)].mean() if (m_ & (p == 0)).any() else pf
        tot += ws.get(s_, 0) * (q * ppv + (1 - q) * fo)
    return tot
rows = []
for p_ in PROBLEMS10:
    pm = L[f'p_{p_}'].values.astype(int); cp = META.loc[COMMON, 'probs'].apply(lambda l, p_=p_: int(p_ in l)).values
    for site in ['medina', 'bahia']:
        r = {'problem': p_, 'site': site, 'LLM_raw_%': cp[csite == site].mean() * 100}
        for ref in ['A1', 'A2']:
            y = hr[f'{ref}_p_{p_}'].values.astype(int)
            c_ = corr_prev_site(y, pm, strata, cp, site); cb = np.array([corr_prev_site(y[i], pm[i], strata[i], cp, site) for i in BI])
            r.update({f'corrected_%_{ref}': c_ * 100, f'corr_lo_{ref}': np.percentile(cb, 2.5) * 100, f'corr_hi_{ref}': np.percentile(cb, 97.5) * 100})
        rows.append(r)
TABLES['R2_corrected_by_site'] = pd.DataFrame(rows).round(3)

# post-stratified kappa: each reference review weighted by (corpus share / sample share) of its star class
def wkappa(a_, b_, quad=False):
    if quad: return cohen_kappa_score(pd.to_numeric(pd.Series(a_)).astype(int), pd.to_numeric(pd.Series(b_)).astype(int), weights='quadratic', sample_weight=wts(strata))
    return cohen_kappa_score(pd.Series(a_).astype(str), pd.Series(b_).astype(str), sample_weight=wts(strata))
rows = []
for d in HDIMS + [f'p_{p}' for p in PROBLEMS10]:
    q = d == 'score'; k1 = wkappa(hr[f'A1_{d}'], L[d], q); k2 = wkappa(hr[f'A2_{d}'], L[d], q); hh = wkappa(hr[f'A1_{d}'], hr[f'A2_{d}'], q)
    rows.append({'dimension': d, 'kappa_A1_w': k1, 'kappa_A2_w': k2, 'kappa_min_w': min(k1, k2), 'kappa_A1_A2_w': hh, 'tier_w': tier(min(k1, k2))})
TABLES['R2_kappa_poststratified'] = pd.DataFrame(rows).round(3)

# (c2) round 3: intervals that remain valid when a cell is empty (Jeffreys posteriors propagated by simulation),
# and error rates estimated within each site (check of the assumption behind the site correction)
NS = 20000; rng = RNG(); site_h = META.loc[hr.index, 'site'].values

def beta_draw(k, n, size):            # Jeffreys posterior for a proportion
    return rng.beta(k + 0.5, n - k + 0.5, size)

# ---------- post-stratified precision / recall: Dirichlet(Jeffreys) per stratum ----------
def wPR_point(y, p):
    tp = fp = fn = 0.0
    for s in LAB:
        m = strata == s; n = m.sum(); w = Wc[s]
        tp += w * ((y == 1) & (p == 1) & m).sum() / n; fp += w * ((y == 0) & (p == 1) & m).sum() / n; fn += w * ((y == 1) & (p == 0) & m).sum() / n
    return tp / (tp + fp) if tp + fp else np.nan, tp / (tp + fn) if tp + fn else np.nan
def wPR_sim(y, p):
    TP = FP = FN = 0
    for s in LAB:
        m = strata == s
        c = np.array([((y == 1) & (p == 1) & m).sum(), ((y == 0) & (p == 1) & m).sum(), ((y == 1) & (p == 0) & m).sum(), ((y == 0) & (p == 0) & m).sum()])
        d = rng.dirichlet(c + 0.5, NS); TP = TP + Wc[s] * d[:, 0]; FP = FP + Wc[s] * d[:, 1]; FN = FN + Wc[s] * d[:, 2]
    return TP / (TP + FP), TP / (TP + FN)

# ---------- corrected prevalence: Beta(Jeffreys) PPV_s and FOR_s, propagated by simulation ----------
def comps(y, p, s, mask=None):
    m = (strata == s) if mask is None else (strata == s) & mask
    return ((y == 1) & (p == 1) & m).sum(), ((p == 1) & m).sum(), ((y == 1) & (p == 0) & m).sum(), ((p == 0) & m).sum()
def corr(y, p, cp, site=None, sitemask_ref=None, sim=False):
    if site is None: wmix = Wc; cm = np.ones(len(cp), bool)
    else: cm = csite == site; wmix = pd.Series(cst[cm]).value_counts(normalize=True).reindex(LAB).fillna(0)
    a_all, n1_all, b_all, n0_all = ((y == 1) & (p == 1)).sum(), (p == 1).sum(), ((y == 1) & (p == 0)).sum(), (p == 0).sum()
    tot = 0
    for s in LAB:
        q = cp[cm & (cst == s)].mean()
        a, n1, b, n0 = comps(y, p, s, sitemask_ref)
        if n1 == 0: a, n1 = a_all, n1_all
        if n0 == 0: b, n0 = b_all, n0_all
        if sim: ppv, fo = beta_draw(a, n1, NS), beta_draw(b, n0, NS)
        else: ppv, fo = a / n1, b / n0
        tot = tot + wmix[s] * (q * ppv + (1 - q) * fo)
    return tot

def ci(sim, pt, scale=1.0):
    lo, hi = np.nanpercentile(sim, [2.5, 97.5])
    # Brown-Cai-DasGupta convention: when a component lies on the boundary the interval is extended to include the point estimate
    return min(lo, pt) * scale, max(hi, pt) * scale
rows, rows_site, rows_for = [], [], []
for pr in PROBLEMS10:
    pm = L[f'p_{pr}'].values.astype(int)
    cp = META.loc[COMMON, 'probs'].apply(lambda l, pr=pr: int(pr in l)).values
    r = {'problem': pr, 'LLM_raw_%': cp.mean() * 100}
    for ref in ['A1', 'A2']:
        y = hr[f'{ref}_p_{pr}'].values.astype(int)
        P, R = wPR_point(y, pm); Ps, Rs = wPR_sim(y, pm)
        c0 = corr(y, pm, cp); cs = corr(y, pm, cp, sim=True)
        (pl, ph), (rl, rh), (cl, chh) = ci(Ps, P), ci(Rs, R), ci(cs, c0, 100)
        r.update({f'Pw_{ref}': P, f'Pw_lo_{ref}': pl, f'Pw_hi_{ref}': ph, f'Rw_{ref}': R, f'Rw_lo_{ref}': rl, f'Rw_hi_{ref}': rh,
                  f'corr_{ref}': c0 * 100, f'corr_lo_{ref}': cl, f'corr_hi_{ref}': chh})
        for site in ['medina', 'bahia']:
            c0s = corr(y, pm, cp, site=site); css = corr(y, pm, cp, site=site, sim=True)
            # sensitivity: error rates estimated on the reference reviews of the same site only
            c1s = corr(y, pm, cp, site=site, sitemask_ref=(site_h == site)); c1ss = corr(y, pm, cp, site=site, sitemask_ref=(site_h == site), sim=True)
            rows_site.append({'problem': pr, 'site': site, 'ref': ref, 'raw_%': cp[csite == site].mean() * 100,
                              'corr_pooled_%': c0s * 100, 'lo': ci(css, c0s, 100)[0], 'hi': ci(css, c0s, 100)[1],
                              'corr_sitespecific_%': c1s * 100, 'lo_ss': ci(c1ss, c1s, 100)[0], 'hi_ss': ci(c1ss, c1s, 100)[1]})
            m0 = (site_h == site) & (pm == 0); k = int(((y == 1) & m0).sum()); n = int(m0.sum())
            lo_, hi_ = (np.percentile(beta_draw(k, n, NS), [2.5, 97.5]) if n else (np.nan, np.nan))
            rows_for.append({'problem': pr, 'site': site, 'ref': ref, 'misses': k, 'non_detections': n, 'FOR_%': 100 * k / n if n else np.nan,
                             'FOR_lo_%': 100 * lo_, 'FOR_hi_%': 100 * hi_, 'ref_positives': int((y[site_h == site] == 1).sum())})
    rows.append(r)
# check: post-stratified accuracy (Table A11) with Jeffreys Beta per star class vs the bootstrap
ra = []
for ref in ['A1', 'A2']:
    yv = hr[f'{ref}_sentiment'].values
    for m_, v in PB.items():
        v = np.asarray(v); pt = 0; sim = 0; zero = False
        for s_ in LAB:
            m = strata == s_; k = int((yv[m] == v[m]).sum()); n = int(m.sum()); zero |= k in (0, n)
            pt += Wc[s_] * k / n; sim = sim + Wc[s_] * beta_draw(k, n, NS)
        lo, hi = ci(sim, pt); ra.append({'ref': ref, 'method': m_, 'Acc_w': pt, 'lo_J': lo, 'hi_J': hi, 'boundary_cell': zero})
A = pd.DataFrame(ra)
TABLES['R3_corrected_jeffreys'] = pd.DataFrame(rows).round(3)
TABLES['R3_site_corrected'] = pd.DataFrame(rows_site).round(2)
TABLES['R3_FOR_by_site'] = pd.DataFrame(rows_for).round(2)
TABLES['R3_A11_jeffreys_check'] = A.round(3)

# (d) per-emotion precision and recall, atmosphere mention rate, validation-sample length coverage
rows = []
for e in EMOTIONS:
    l_ = (L.emotion == e).astype(int).values; r = {'emotion': e, 'LLM_n_sample': int(l_.sum()), 'corpus_%': (df.llm_emotion == e).mean() * 100}
    for ref in ['A1', 'A2']:
        y = (hr[f'{ref}_emotion'] == e).astype(int).values; p_, r_, f_, _ = precision_recall_fscore_support(y, l_, average='binary', zero_division=0)
        r.update({f'n_{ref}': int(y.sum()), f'P_{ref}': p_, f'R_{ref}': r_, f'F1_{ref}': f_})
    rows.append(r)
TABLES['R2_emotion_per_class'] = pd.DataFrame(rows).round(3)
TABLES['R2_atmosphere_mention'] = pd.DataFrame([{'source': nm, 'mention_rate': (np.asarray(v) != 'Not_mentioned').mean()} for nm, v in
    [('LLM (300)', L.asp_atmosphere.values), ('annotator 1', hr.A1_asp_atmosphere.values), ('annotator 2', hr.A2_asp_atmosphere.values),
     ('LLM (corpus)', df.llm_atmosphere.values)]]).round(3)
ln = META.text_len_chars
TABLES['R2_length_coverage'] = pd.DataFrame([{'n_lt80': int((ln < 80).sum()), 'pct_lt80': (ln < 80).mean() * 100, 'pct_80_1200': ((ln >= 80) & (ln <= 1200)).mean() * 100,
                                             'pct_gt1200': (ln > 1200).mean() * 100, 'median': ln.median(), 'n_gt2000': int((ln > 2000).sum())}]).round(2)

# (e) problem-emotion association: permutation test that resamples reviews (emotions shuffled across reviews)
sub = df[df.probs.apply(lambda x: any(p in PROBLEMS10 for p in x))]
emo = sub.llm_emotion.astype(str).values; plist = [[p for p in dict.fromkeys(x) if p in PROBLEMS10] for x in sub.probs]
def chi_stat(e):
    a = [p for ps in plist for p in ps]; b = [ee for ps, ee in zip(plist, e) for _ in ps]
    t = pd.crosstab(pd.Series(a), pd.Series(b)); return chi2_contingency(t)[0], t
obs, t_ = chi_stat(emo); rng = RNG(); null = np.array([chi_stat(rng.permutation(emo))[0] for _ in range(1000)])
TABLES['R2_permutation_test'] = pd.DataFrame([{'reviews': len(sub), 'pairs': int(t_.values.sum()), 'chi2_observed': obs,
    'cramers_V': np.sqrt(obs / t_.values.sum() / (min(t_.shape) - 1)), 'p_permutation': (1 + (null >= obs).sum()) / 1001, 'n_permutations': 1000,
    'null_95th': np.percentile(null, 95)}]).round(4)

# (f) original configuration vs main analysis (corpus level)
OG = orig_outputs(COMMON); X1 = RUNS['run1'].loc[COMMON]
cmp_ = {}
for nm, X in [('original configuration', OG), ('main analysis (run 1)', X1)]:
    xs = X.sentiment.astype(str).where(X.sentiment.astype(str).isin(LAB), 'Neutral')
    cmp_[nm] = {**{f'{l_}%': (xs == l_).mean() * 100 for l_ in LAB}, **{f'{p_}%': X[f'p_{p_}'].mean() * 100 for p_ in PROBLEMS10},
                **{f'Neg%_{per}': (xs.values[(META.loc[COMMON, 'period'] == per).values] == 'Negative').mean() * 100 for per in PERIODS},
                'kappa_vs_main_sentiment': kappa(xs, X1.sentiment)}
TABLES['R2_original_vs_main'] = pd.DataFrame(cmp_).round(2).reset_index()

# %% [markdown]
# ## 12. Save everything

# %%
with pd.ExcelWriter(f'{OUT}/all_tables.xlsx') as xw:
    for name, t in TABLES.items(): t.to_excel(xw, sheet_name=name[:31], index=False)
print('Saved', len(TABLES), 'tables to', f'{OUT}/all_tables.xlsx', 'and figures 3-8 to', OUT)
