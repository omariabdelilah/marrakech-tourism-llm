# marrakech-tourism-llm

Data and code for the paper **"A single-pass, training-free large language model framework for multidimensional analysis of tourism reviews"** (Abdelilah Omari and Imane Satauri, L3IA Laboratory, Sidi Mohamed Ben Abdellah University, Fez, Morocco).

The repository lets anyone regenerate every table (1–11, A1–A8) and figure (3–8) of the paper **without API calls and without the review texts**.

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook reproduce_all.ipynb     # or open it in Google Colab together with the data/ folder
```

`reproduce_all.ipynb` writes to `outputs/`:

* `all_tables.xlsx`: one sheet per table.
* `fig3_confusion.png` … `fig8_heatmap.png`: the figures, redrawn from the data. The layout may differ slightly from the published figures, but the numbers are the same.

Bootstrap confidence intervals use 2,000 resamples with seed 42 (1,000 for Table A5). Because each section starts its own random generator, the last digit of a confidence interval can differ from the paper.

## Contents

| Path | Content |
|---|---|
| `reproduce_all.ipynb` (and `.py`) | Preprocessing and all analyses of the paper, from the files in `data/`. |
| `prompt.txt` | The complete prompt sent to the models: schema, rules and the two format examples (Figure 1). |
| `notebooks/gemini_original_run.ipynb` | Original Gemini 2.5 Flash run on the 14,838 reviews. |
| `notebooks/baselines_vader_bert.ipynb` | VADER and BERT baselines, and language identification. |
| `notebooks/gemini_robustness_runs.ipynb` | Robustness runs of Section 3.6: truncation, joint versus dedicated prompts, and run-to-run variability. |
| `notebooks/gemini_translation_check.ipynb` | Translation-fidelity check on 133 original-language reviews. |
| `data/corpus_llm_outputs.csv` | One row per review (14,838): anonymised index, site, star rating, date, trip type, text length and all Gemini outputs of the original run. |
| `data/baselines_vader_bert.csv` | VADER and BERT predictions on the 500-character and full texts. |
| `data/human_reference_300.csv` | Labels of annotator 1 (`A1_*`) and annotator 2 (`A2_*`) for the 300 reference reviews. |
| `data/interllm_gemini_claude_300.csv` | Gemini (`G_*`) and Claude Opus 4.7 (`C_*`) outputs for the inter-LLM sample. |
| `data/reruns/` | Raw JSON outputs, token logs and run information of the robustness runs (23 September 2026). |
| `data/translation/` | Outputs on the English (`EN`, `EN2`) and original (`ORIG`) texts, sample description, token log (24 September 2026). |

## Data notes

* **No review text and no user name is redistributed**, in accordance with TripAdvisor's terms of service. Reviews are identified by `row_id`, their position in the collected file. The same `row_id` is used in every file, including `data/reruns/*` and `data/translation/*`.
* **Preprocessing.** Review texts were analysed as collected. The only processing was trimming leading and trailing whitespace, and truncating to the first 500 characters for the main analysis; this is the same input for Gemini, VADER and BERT. No reviews were removed; the corpus contains 10 exact duplicate records, which were kept. Star ratings are mapped to three classes: 4–5 Positive, 3 Neutral, 1–2 Negative.
* **Human reference sample.** 100 reviews were randomly sampled from each star-derived class. The sampled IDs are the `row_id` values in `data/human_reference_300.csv`. Both annotators labelled the full review text independently, with the same label definitions.
* **Models and settings.** `gemini-2.5-flash` via `google-generativeai` 0.8.5: batches of 15 reviews, 4 parallel requests, up to 3 retries, provider default generation settings (no temperature set). Outputs of commercial models may change between runs and model versions; the measured run-to-run variability is reported in Section 4.9 of the paper.
* **API keys** are typed at run time with `getpass` and are never stored in the notebooks.

## Licence

Code: MIT (see `LICENSE`). Derived data: CC BY 4.0.
