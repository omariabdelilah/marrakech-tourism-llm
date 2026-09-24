# %% [markdown]
# # Reproduction notebook
# **A single-pass, training-free large language model framework for multidimensional analysis of tourism reviews**
#
# This notebook regenerates every table (1–11, A1–A8) and figure (3–8) of the paper from the
# anonymised files in `data/`. It makes **no API calls** and needs no review text.
#
# Outputs (folder `outputs/`):
# * `all_tables.xlsx` – one sheet per table;
# * `fig3_confusion.png` … `fig8_heatmap.png`.
#
# Bootstrap confidence intervals use 2,000 resamples with seed 42 (1,000 for Table A5). Because each
# section starts its own random generator, the last digit of a CI can differ slightly from the paper.

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
# The 16 off-schema "Mixed" sentiment outputs are grouped with Neutral (as in Figure 3).

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
bl = pd.read_csv(f'{DATA}/baselines_vader_bert.csv')
assert (df.row_id == bl.row_id).all()
df = df.merge(bl, on='row_id')
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

def tier(k): return 'High' if k >= 0.65 else ('Moderate' if k >= 0.40 else 'Low')

# %% [markdown]
# ## 3. Table 1 (Panel A), Table 2, Table A1 and Figure 3 — polarity against star-derived labels

# %%
t1a, tA1 = compare_methods(df['star_label'], {'VADER': df['vader_trunc'], 'BERT': df['bert_trunc'], 'LLM': df['llm_pred']})
for m, col in [('VADER', 'vader_trunc'), ('BERT', 'bert_trunc'), ('LLM', 'llm_pred')]:
    for av in ['macro', 'weighted']:
        p, r, f, _ = precision_recall_fscore_support(df.star_label, df[col], labels=LAB, average=av, zero_division=0)
        t1a.loc[t1a.Method == m, f'{av[0]}P'] = p; t1a.loc[t1a.Method == m, f'{av[0]}R'] = r
        t1a.loc[t1a.Method == m, f'{av[0]}F1'] = f
TABLES['Table1_PanelA'] = t1a.round(3); TABLES['TableA1_paired'] = tA1.round(4)
print(t1a.round(3).to_string(index=False)); print(tA1.round(3).to_string(index=False))

# Sensitivity on untruncated input (footnote of Table 1)
sens, _ = compare_methods(df['star_label'], {'VADER full': df['vader_full'], 'BERT full': df['bert_full']})
TABLES['Table1_untruncated_sensitivity'] = sens.round(3)

# Table 2: per-class, LLM
p, r_, f, s = precision_recall_fscore_support(df.star_label, df.llm_pred, labels=LAB, zero_division=0)
t2 = pd.DataFrame({'Class': LAB, 'Support': s, 'Precision': p, 'Recall': r_, 'F1': f}).round(3)
TABLES['Table2_LLM_per_class'] = t2; print(t2.to_string(index=False))

# Figure 3
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, (m, col) in zip(ax, [('VADER', 'vader_trunc'), ('BERT', 'bert_trunc'), ('LLM', 'llm_pred')]):
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

t1b, pb = compare_methods(hr['A1_sentiment'], {'VADER': S.vader_trunc.values, 'BERT': S.bert_trunc.values,
                                               'LLM': L.sentiment.values, 'Star-derived labels': S.star_label.values})
for a in ['A2']:
    for m, v in [('VADER', S.vader_trunc), ('BERT', S.bert_trunc), ('LLM', L.sentiment), ('Star-derived labels', S.star_label)]:
        t1b.loc[t1b.Method == m, 'Acc_vs_A2'] = (v.values == hr[f'{a}_sentiment'].values).mean()
cons = hr['A1_sentiment'] == hr['A2_sentiment']
for m, v in [('VADER', S.vader_trunc), ('BERT', S.bert_trunc), ('LLM', L.sentiment), ('Star-derived labels', S.star_label)]:
    t1b.loc[t1b.Method == m, 'Acc_consensus'] = (v.values[cons.values] == hr['A1_sentiment'].values[cons.values]).mean()
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
    k1, lo, hi = kappa_ci(a1, l, w)
    r = {'dimension': d, 'kappa_A1': k1, 'CI_lo': lo, 'CI_hi': hi, 'kappa_A2': kappa(a2, l, w), 'kappa_A1_A2': kappa(a1, a2, w),
         'accuracy_A1': (a1.astype(str).values == l.astype(str).values).mean()}
    if w: r['within_1'] = (abs(a1.astype(int).values - l.astype(int).values) <= 1).mean()
    else:
        for av in ['macro', 'weighted']:
            r[f'{av}_F1'] = precision_recall_fscore_support(a1.astype(str), l.astype(str), average=av, zero_division=0)[2]
    if d.startswith('p_'):
        pp, rr, ff, _ = precision_recall_fscore_support(a1.astype(int), l.astype(int), average='binary', zero_division=0)
        r.update({'support': int(a1.sum()), 'P': pp, 'R': rr, 'F1': ff})
    r['tier_point'] = tier(k1); r['CI_crosses_boundary'] = tier(lo) != tier(hi)
    rows.append(r)
t78 = pd.DataFrame(rows)
TABLES['Table7_8_human_validation'] = t78.round(3); print(t78.round(3).to_string(index=False))

# Identical-input subset and 3-problem cap
short = S.text_len_chars.values <= TRUNC
cap = hr[[f'A1_p_{p}' for p in PROBLEMS10]].sum(1).values <= 3
rows = []
for d in HDIMS + [f'p_{p}' for p in PROBLEMS10]:
    w = 'quadratic' if d == 'score' else None
    rows.append({'dimension': d, 'kappa_identical_input': kappa(hr[f'A1_{d}'][short], L[d][short], w),
                 'kappa_excl_>3_problems': kappa(hr[f'A1_{d}'][cap], L[d][cap], w)})
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
ax[0].barh(top8.index[::-1], (top8 / N * 100)[::-1], color='#4C72B0'); ax[0].set_xlabel('% of reviews'); ax[0].set_title('(a) Primary emotional tone')
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
il = pd.read_csv(f'{DATA}/interllm_gemini_claude_300.csv')
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
pairs = pd.DataFrame([(p, e) for x, e in zip(df.probs, df.llm_emotion) for p in x if p in PROBLEMS10 and e in EMOTIONS],
                     columns=['problem', 'emotion'])
ct = pd.crosstab(pairs.problem, pairs.emotion); c2, p, dof, V = cramers_v(ct)
# Table 11 and Figure 8: P(emotion | problem) over all emotion labels (off-schema labels stay in the denominator)
pairs_all = pd.DataFrame([(p, e) for x, e in zip(df.probs, df.llm_emotion) for p in x if p in PROBLEMS10], columns=['problem', 'emotion'])
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
    'jaccard_lo': np.percentile(jb, 2.5), 'jaccard_hi': np.percentile(jb, 97.5), 'pearson_r_100_cells': pearsonr(x, y_)[0],
    'spearman_rho': spearmanr(x, y_)[0], 'same_emotion_when_both_flag': (hr.A1_emotion.values[anyboth] == L.emotion.values[anyboth]).mean()}]).round(3)

# %% [markdown]
# ## 10. Section 4.9 — robustness runs (Tables A3, A5, A6, A7) and cost

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
def orig_outputs(ids):
    s = df.set_index('row_id').loc[ids]
    o = pd.DataFrame({'sentiment': s.llm_sentiment.values, 'score': s.llm_score.values, 'emotion': s.llm_emotion.values,
                      'price_perception': s.llm_price_perception.values, 'recommendation': s.llm_recommendation.values,
                      'revisit': s.llm_revisit.values, 'problems': s.probs.values}, index=ids)
    for a in ASPECTS: o[f'asp_{a}'] = s[f'llm_{a}'].values
    return addp(o)
R = f'{DATA}/reruns'
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

# Table A5: truncation (same-day runs on long reviews)
AT, AF = load_run(f'{R}/partA_trunc.csv'), load_run(f'{R}/partA_full.csv'); LONG = list(AT.index)
AT, AF = addp(AT.loc[LONG]), addp(AF.loc[LONG])
def vec(x):
    s = x.sentiment.astype(str).replace({'Mixed': 'Neutral'})
    o = {'>=1 problem %': x.problems.apply(lambda l: len([p for p in (l or []) if p in PROBLEM_SCHEMA]) > 0).values * 100.,
         'Scam %': x.p_Scam.values * 100., 'Aggressive vendors %': x.p_Aggressive_vendors.values * 100., 'Safety problem %': x.p_Safety.values * 100.,
         'High prices %': x.p_High_prices.values * 100., 'Positive %': s.eq('Positive').values * 100., 'Negative %': s.eq('Negative').values * 100.,
         'Mean number of problems': x.problems.apply(lambda l: len(l or [])).values * 1.}
    return o
vt, vf = vec(AT), vec(AF); rng = RNG(); idx = rng.integers(0, len(LONG), (1000, len(LONG))); rows = []
for k_ in vt:
    d_ = vf[k_] - vt[k_]; bs = d_[idx].mean(1)
    rows.append({'indicator': k_, 'truncated': vt[k_].mean(), 'full_text': vf[k_].mean(), 'diff': d_.mean(), 'lo': np.percentile(bs, 2.5), 'hi': np.percentile(bs, 97.5)})
tA5 = pd.DataFrame(rows); TABLES['TableA5_truncation'] = tA5.round(2); print(tA5.round(2).to_string(index=False))
# corpus-level effect: original outputs for short reviews + same-day outputs for long reviews
base = orig_outputs([i for i in df.row_id if i not in set(LONG)])
V1, V2 = pd.concat([base, AT]).loc[df.row_id], pd.concat([base, AF]).loc[df.row_id]
ce = pd.DataFrame({'long truncated': [V1[f'p_{p}'].mean() * 100 for p in PROBLEM_SCHEMA], 'long full text': [V2[f'p_{p}'].mean() * 100 for p in PROBLEM_SCHEMA]}, index=PROBLEM_SCHEMA)
ce['diff'] = ce.iloc[:, 1] - ce.iloc[:, 0]; TABLES['Sec4_9_corpus_effect'] = ce.round(2).reset_index()
# 57 long reviews of the human sample
L57 = [i for i in IDS if i in set(LONG)]

# Table A7: translation fidelity
T = f'{DATA}/translation'
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
cost.loc[len(cost)] = ['TOTAL (token logs)', summ.cost_usd.sum()]
TABLES['Sec3_3_cost'] = cost

# %% [markdown]
# ## 11. Table A4 — validity evidence behind each management implication

# %%
kk = t78.set_index('dimension')
fmt = lambda d: f"{kk.loc[d, 'kappa_A1']:.3f} [{kk.loc[d, 'CI_lo']:.3f}, {kk.loc[d, 'CI_hi']:.3f}]"
TABLES['TableA4_implications'] = pd.DataFrame([
    ['R1 Commercial actors (Medina)', 'Aggressive vendors; emotional tone', f"{fmt('p_Aggressive_vendors')}; {fmt('emotion')}", 'Supported'],
    ['R2 Restoration and interpretation (Bahia)', 'Poor condition', fmt('p_Poor_condition'), 'Indicative'],
    ['R3 2024-2025 deterioration', 'Sentiment; overcrowding; high prices; poor condition',
     f"{fmt('sentiment')}; {kk.loc['p_Overcrowding','kappa_A1']:.3f}; {kk.loc['p_High_prices','kappa_A1']:.3f}; {kk.loc['p_Poor_condition','kappa_A1']:.3f}",
     'Supported (sentiment); indicative (drivers)'],
    ['R4 Safety (Medina)', 'Safety problem; Fear', f"{fmt('p_Safety')}; {kk.loc['emotion','kappa_A1']:.3f}", 'Supported']],
    columns=['Implication', 'Dimensions used', 'kappa [95% CI]', 'Status'])

# %% [markdown]
# ## 12. Save everything

# %%
with pd.ExcelWriter(f'{OUT}/all_tables.xlsx') as xw:
    for name, t in TABLES.items(): t.to_excel(xw, sheet_name=name[:31], index=False)
print('Saved', len(TABLES), 'tables to', f'{OUT}/all_tables.xlsx', 'and figures 3-8 to', OUT)
