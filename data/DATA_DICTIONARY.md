# Data dictionary — `llm_extracted_signals.csv`

14,838 rows (one per review), 22 columns. Collected from TripAdvisor for two
heritage attractions in Marrakech, Morocco, covering 2004 to early 2026.

Reviewer names, review titles, review bodies and contribution counts are **not**
included: review texts are not redistributed in accordance with TripAdvisor's
terms of service, and all reviewer identifiers were removed prior to analysis.

---

## Identification and metadata

| Column | Type | Description |
|---|---|---|
| `review_id` | integer | Sequential identifier, 1–14,838. Arbitrary; carries no meaning beyond row identity. |
| `place` | string | `medina` (n = 8,940) or `bahia` (n = 5,898). |
| `stars` | float | Star rating, 1–5, parsed from the platform's rating string. |
| `gt` | string | Three-class mapping of `stars` used as ground truth throughout the paper: 4–5 → `Positive`, 3 → `Neutral`, 1–2 → `Negative`. |
| `year` | integer | Year the review was written, parsed from `written_date`. |
| `era` | string | `Pre-COVID` (≤ 2019, n = 12,708), `COVID` (2020, n = 555), `Post-COVID` (≥ 2021, n = 1,575). |
| `trip_date` | string | Month and year of the visit, as reported by the reviewer (e.g. `Mar 2026`). |
| `trip_type` | string | `Couples`, `Friends`, `Solo`, `Business`, or empty. Available for 10,725 reviews (72.3%). |
| `written_date` | string | Publication date as displayed by the platform. |

## Fields extracted by the LLM

All thirteen fields below were produced by Gemini 2.5 Flash in a single
zero-shot inference per review, under the prompt in
`code/extraction_prompt.txt`.

| Column | Vocabulary | Description |
|---|---|---|
| `llm_sentiment` | `Positive`, `Neutral`, `Negative` | Overall stance. 16 reviews (0.11%) carry the off-schema value `Mixed`; the paper groups these with `Neutral`. |
| `llm_score` | 1–5 | Sentiment intensity. |
| `llm_emotion` | `Joy`, `Anger`, `Fear`, `Disgust`, `Sadness`, `Surprise`, `Trust`, `Frustration`, `Neutral`, `Other` | Primary emotional tone. 352 reviews (2.4%) carry labels outside this vocabulary, most frequently `Disappointment` (n = 190). |
| `llm_safety` | `Positive`, `Negative`, `Neutral`, `Not_mentioned` | Aspect-level sentiment. |
| `llm_pricing` | as above | Aspect-level sentiment. 35 off-schema values (0.24%). |
| `llm_service` | as above | Aspect-level sentiment. 6 off-schema values (0.04%). **Low-validity tier** (κ = 0.174 against the human gold standard): the model assigns vendor-harassment content to this aspect, whereas the annotator reserved it for formal hospitality. |
| `llm_cleanliness` | as above | Aspect-level sentiment. |
| `llm_atmosphere` | as above | Aspect-level sentiment. 65 off-schema values (0.44%). |
| `llm_problems` | JSON list | Zero to three problems drawn from an eleven-category taxonomy: `Aggressive_vendors`, `Navigation_difficulty`, `Getting_lost`, `Bad_smell`, `Overcrowding`, `Poor_condition`, `High_prices`, `Poor_cleanliness`, `Scam`, `Safety`, `Other`. A further 906 mentions across 303 distinct strings fall outside the taxonomy. |
| `llm_price_perception` | `Expensive`, `Fair`, `Cheap`, `Not_mentioned` | Economic signal. 10 off-schema values (0.07%). |
| `llm_recommendation` | `Yes`, `No`, `Not_mentioned` | Economic signal. 129 off-schema values (0.87%). **Low-validity tier** (κ = 0.343). |
| `llm_revisit` | `Yes`, `No`, `Not_mentioned` | Economic signal. **Low-validity tier** (κ = 0.316). |
| `llm_keywords` | JSON list | Three to five salient terms per review, translated to English by the model. |

## Reading the extracted fields

Reliability is not uniform across dimensions. The paper's human gold-standard
validation on an independent stratified sample of 300 reviews partitions them as
follows, and anyone reusing this dataset should apply the same distinction:

- **High validity** (κ ≥ 0.65) — `llm_sentiment`, `llm_score`, `llm_safety`,
  `llm_cleanliness`, `llm_pricing`, `llm_price_perception`, and the problem
  categories with concrete lexical triggers (`Aggressive_vendors`, `Scam`,
  `Bad_smell`, `Overcrowding`, `High_prices`, `Safety`).
- **Moderate** (0.40 ≤ κ < 0.65) — `llm_emotion`, `llm_atmosphere`, and the
  `Poor_condition` and `Getting_lost` problem categories.
- **Low validity** (κ < 0.40) — `llm_service`, `llm_recommendation`,
  `llm_revisit`, and the `Navigation_difficulty` and `Other` problem categories.
  Findings drawn from these are exploratory.

## Two further properties worth knowing

**Class imbalance.** The corpus is dominated by positive reviews (79.6%
`Positive` under `gt`) and by the pre-pandemic period (86% of rows). Any model
trained or evaluated on this dataset should account for both.

**Machine translation.** Reviews were contributed by an international visitor
base and retrieved with the platform's translation feature enabled, so all texts
were processed in English regardless of the language in which they were written.
The original-language tag was not retained by the scraping pipeline. The
extracted signals therefore characterise a translated corpus.
