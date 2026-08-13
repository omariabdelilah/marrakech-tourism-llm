# marrakech-tourism-llm

Data and code for the paper:

> **A Single-Inference Zero-Shot LLM Framework for Multi-Dimensional Analysis of Machine-Translated Multilingual Tourism Reviews**

The paper presents a framework based on Large Language Models that extracts six
analytical dimensions — sentiment polarity, sentiment intensity, emotional tone,
aspect-level opinions, reported problems, and economic signals — from a single
zero-shot inference, and applies it to 14,838 TripAdvisor reviews of two heritage
attractions in Marrakech, Morocco (the Medina and Bahia Palace).

This repository releases the extracted-signal dataset and the analysis code, so
that the reported distributions, cross-site contrasts, temporal trends and
cross-dimensional statistics can be recomputed from scratch.

---

## Repository layout

```
marrakech-tourism-llm/
├── data/
│   ├── llm_extracted_signals.csv   14,838 rows — the released dataset
│   └── DATA_DICTIONARY.md          column-by-column description
├── code/
│   ├── 00_prepare_data.py          builds the released dataset (see note below)
│   ├── 01_analysis.py              reproduces the tables and statistics
│   ├── 02_figures.py               reproduces the figures
│   ├── 03_baselines.py             VADER and BERT baselines (see note below)
│   └── extraction_prompt.txt       the zero-shot prompt, verbatim as executed
├── outputs/                        written by the scripts
├── requirements.txt
├── LICENSE                         MIT — applies to the code
└── LICENSE-DATA                    CC BY 4.0 — applies to data/
```

## Quick start

```bash
git clone https://github.com/<user>/marrakech-tourism-llm.git
cd marrakech-tourism-llm
pip install -r requirements.txt

python code/01_analysis.py     # tables and statistics -> outputs/
python code/02_figures.py      # figures -> outputs/
```

Both scripts read only `data/llm_extracted_signals.csv` and take under a minute
on a laptop. No API key, no GPU and no network access are required.

## What the scripts reproduce

`01_analysis.py` recomputes:

| Output | Content |
|---|---|
| Table 3 | Aspect-based sentiment distribution, with off-schema label counts |
| Table 4 | Top reported problems, with the free-form label audit |
| Table 5 | Economic signals, with off-schema label counts |
| Table 9 | Problem mention rates by heritage site, with Medina/Bahia ratios |
| Table 10 | Sentiment distribution by era, plus the year-by-year series |
| Table 11 | Conditional emotion probabilities given the four most frequent problems |
| Section 4.6 | Cross-site emotional, aspect-level and trip-type contrasts |
| Section 4.7 | Pre- versus post-pandemic problem shifts, including the within-site checks |
| Section 4.8 | Chi-square test of independence and Cramér's *V* |

`02_figures.py` regenerates Figures 4, 5, 6, 7 and 8 as PNG (600 dpi) and PDF.

Every table is also written to `outputs/` as CSV.

## Two scripts that cannot be re-run here

`00_prepare_data.py` and `03_baselines.py` require the raw scrape, which is not
part of this release: **review texts are not redistributed, in accordance with
TripAdvisor's terms of service**, and all reviewer identifiers were removed prior
to analysis. Both scripts are included so that the de-identification steps and
the baseline configuration — in particular the input-harmonization procedure of
Section 3.3 — can be inspected in full.

## The dataset

`data/llm_extracted_signals.csv` contains one row per review and 22 columns: the
star rating and its three-class mapping, the site, the review year and pandemic
era, the trip metadata, and the thirteen fields extracted by the LLM. Reviewer
names, review titles and review bodies are not included. See
[`data/DATA_DICTIONARY.md`](data/DATA_DICTIONARY.md) for the full column
description and the label vocabulary of each field.

A caveat carried over from the paper: reliability is not uniform across the
extracted dimensions. The human gold-standard validation partitions them into
high-, moderate- and low-validity tiers, and figures drawn from the low tier —
the service aspect, recommendation and revisit intention — are reported as
exploratory. Anyone reusing this dataset should apply the same distinction.

## Extraction configuration

Extraction used Gemini 2.5 Flash at temperature 0 with structured JSON output,
under the prompt reproduced verbatim in `code/extraction_prompt.txt`. Reviews
were truncated to 500 characters before processing. Because models served
through commercial APIs are updated over time, re-running the extraction at a
later date may not reproduce these labels exactly; the extracted labels are
released here for that reason.

## Licence

Code is released under the MIT Licence. The dataset in `data/` is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

See [`CITATION.cff`](CITATION.cff).
