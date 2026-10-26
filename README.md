# marrakech-tourism-llm

Data and code for the paper **"A training-free large language model framework for multidimensional analysis of tourism reviews"** (Abdelilah Omari and Imane Satauri, L3IA Laboratory, Sidi Mohamed Ben Abdellah University, Fez, Morocco; submitted to *Discover Artificial Intelligence*).

The repository lets anyone regenerate every table (1–11, A1–A12) and figure (3–8) of the paper **without API calls and without the review texts**.

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook reproduce_all.ipynb     # or: python reproduce_all.py
```

`reproduce_all` writes to `outputs/`:

* `all_tables.xlsx`: one sheet per table, plus the sheets `R2_*` for the robustness analyses (run-to-run variability, batch composition, rubric-aligned prompt, post-stratification, corrected prevalence, per-emotion metrics, permutation test) and `R3_*` for the intervals of the corrected rates and the site-specific error rates.
* `fig3_confusion.png` … `fig8_heatmap.png`: the figures, redrawn from the data (layout may differ slightly from the published figures; the numbers are the same).

Bootstrap confidence intervals use 2,000 resamples with seed 42. The intervals of the post-stratified precision and recall and of the corrected problem rates (Tables 4, 8, 9 and A12) use Jeffreys posteriors propagated by simulation (20,000 draws, seed 42), which remain valid when a cell of the reference sample is empty (sheets `R3_*`, added in the third revision, which also contain the false-omission rates by site). The full script takes about 20 minutes.

## Main configuration

* Model `gemini-2.5-flash` via `google-genai` 2.26.0; temperature 0, seed 42, thinking budget 8,192 tokens.
* Full review text; 15 reviews per request; for each run the reviews are shuffled with a fixed seed (101, 102, 103) and cut into consecutive batches of 15. A batch that still fails after 4 attempts is re-sent one review per request.
* Three full-corpus runs (6–7 October 2026). Per-review results in the paper use run 1; headline quantities are the mean ± SD over the three runs.
* One review (`row_id` 8302) received no valid output in run 1 and is excluded from the analyses of that run (n = 14,837).

## Contents

| Path | Content |
|---|---|
| `reproduce_all.ipynb` (and `.py`) | All analyses of the paper, from the files in `data/`. |
| `prompts/joint_prompt.txt` | The main prompt (six dimensions, two format examples; Figure 1 of the paper). `{REVIEWS_JSON}` is replaced by the JSON array of the batch. |
| `prompts/polarity_only_prompt.txt` | Polarity-only prompt (single-task LLM baseline). |
| `prompts/rubric_aligned_prompt.txt` | Prompt with the annotators' explicit-mention rule and without the format examples. |
| `prompts/original_run_prompt.txt` | The prompt as used in the first version (identical wording, 500-character input). |
| `notebooks/gemini_revision2_runs.ipynb` | Inference for the main analysis: the three main runs, polarity-only, single-review calls (2,000 random + 300 reference reviews) and rubric-aligned runs, with token log and budget guard. |
| `notebooks/siebert_baseline.ipynb` | SiEBERT baseline (`siebert/sentiment-roberta-large-english`) on the full and 500-character texts. |
| `notebooks/baselines_vader_bert.ipynb` | VADER and BERT baselines, and language identification. |
| `notebooks/gemini_original_run.ipynb` | Run of the first version (500 characters, default settings, file-order batches). |
| `notebooks/gemini_robustness_runs.ipynb` | Checks in the preliminary configuration: joint versus dedicated prompts, truncation and same-day re-runs. |
| `notebooks/gemini_translation_check.ipynb` | Cross-language consistency check on 133 original-language reviews. |
| `data/corpus_llm_outputs.csv` | One row per review (14,838): anonymised index, site, star rating, written date, trip type, text length, the Gemini outputs of run 1, and `batch_id` / `pos_in_batch`. |
| `data/runs/` | Raw outputs of the main analysis: `run1.csv`–`run3.csv`, `polarity_only.csv`, `single_call.csv`, `single_300.csv`, `rubric_300.csv`, `token_log.csv` (every request, with tokens and model version), `summary.csv`, `run_info.json`. |
| `data/baselines.csv` | VADER, BERT and SiEBERT predictions on the full and 500-character texts (SiEBERT also with its positive-class probability). |
| `data/human_reference_300.csv` | Labels of annotator 1 (`A1_*`) and annotator 2 (`A2_*`) for the 300 reference reviews. |
| `data/text_fingerprints.csv` | SHA-256 of the UTF-8 review text of every review, to match a re-collected corpus to the released outputs. |
| `data/original_run/` | Outputs of the first version: corpus outputs, inter-LLM sample (Gemini `G_*`, Claude Opus 4.7 `C_*`), the joint-versus-dedicated and truncation re-runs (`reruns/`) and cross-language check (`translation/`). |

## Data notes

* **No review text and no user name is redistributed**, in line with the platform's terms of use; only derived structured outputs are released. Reviews are identified by `row_id`, their position in the collected file, and the same `row_id` is used in every file. The review texts are available from the corresponding author on reasonable request for verification. The collection script is not released; the procedure is described in Section 3.1 of the paper.
* **Collection.** All reviews listed on the public TripAdvisor pages of the two attractions at the end of March 2026, language filter set to all languages (non-English reviews in TripAdvisor's English machine translation), most recent first.
* **Preprocessing.** Review texts were analysed as collected, after trimming leading and trailing whitespace. No reviews were removed; the corpus contains 10 exact duplicate records, which were kept. Star ratings are mapped to three classes: 4–5 Positive, 3 Neutral, 1–2 Negative.
* **Human reference sample.** 100 reviews were randomly sampled from each star-derived class (80–1,200 characters). Both annotators labelled the full review text independently; neither is treated as primary. Reliability tiers use the lower of the two LLM–annotator κ values (Landis–Koch bands: high ≥ 0.61, moderate 0.40 < κ < 0.61, low ≤ 0.40).
* **Reproducibility.** Commercial models may be updated or retired, and even with fixed settings repeated runs differ at the level of individual labels (sentiment κ between runs ≈ 0.84). Reproducibility therefore applies to the released outputs and this analysis pipeline rather than to exact regeneration of individual labels.
* **API keys** are typed at run time with `getpass` and are never stored in the notebooks.

## Licence

Code: MIT (see `LICENSE`). Derived data: CC BY 4.0.
