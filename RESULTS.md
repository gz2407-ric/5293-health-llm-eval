# RESULTS

This document summarizes the current locked-down results for the STAT GR5293 project, **Evaluating LLM Architectures for Personalized Lifestyle Health Recommendations**.

## Experiment Setup

We evaluated four LLM system designs:

| System | RAG | Multi-step pipeline | Description |
|---|---:|---:|---|
| Baseline | No | No | Single-prompt generation |
| RAG only | Yes | No | Retrieves CDC/WHO guideline chunks before generation |
| Multi-step only | No | Yes | Profile parsing, risk flagging, generation, safety check |
| Full pipeline | Yes | Yes | RAG + multi-step generation + safety check |

Data:

- **50 synthetic health-profile scenarios**, stratified into six scenario types.
- **50 CDC/WHO guideline chunks** used as the RAG corpus.
- **200 model outputs** generated from 50 scenarios × 4 systems.
- **800 LLM-as-judge scores** generated from 200 outputs × 4 metrics.

Evaluation metrics:

- Relevance
- Personalization
- Guideline Alignment
- Safety

## 1. Main LLM-as-Judge Results

| system_name     |   Relevance |   Personalization |   Guideline Alignment |   Safety |
|:----------------|------------:|------------------:|----------------------:|---------:|
| Baseline        |        4.00 |              4.08 |                  3.06 |     4.28 |
| RAG only        |        4.00 |              4.00 |                  4.72 |     4.18 |
| Multi-step only |        4.00 |              4.08 |                  3.06 |     4.48 |
| Full pipeline   |        4.10 |              4.12 |                  4.84 |     4.28 |

### Main Finding

The strongest result is that **retrieval-augmented systems substantially improved guideline alignment**. Baseline and Multi-step-only systems scored around **3.06**, while RAG-only and Full pipeline scored **4.72** and **4.84**, respectively.

The Full pipeline achieved the best overall balance, with the highest scores in Relevance, Personalization, and Guideline Alignment, but it did not dominate Safety. Multi-step-only achieved the highest Safety score, suggesting more conservative risk-aware behavior.

## 2. Statistical Significance

We used paired Wilcoxon signed-rank tests across scenarios, followed by Benjamini-Hochberg FDR correction. Significant results at adjusted p < 0.05 are summarized below.

| metric              | comparison                       |   mean_difference_b_minus_a |   p_value |   adjusted_p_value_fdr_bh |   effect_size_rank_biserial |
|:--------------------|:---------------------------------|----------------------------:|----------:|--------------------------:|----------------------------:|
| Guideline Alignment | Baseline vs RAG only             |                        1.66 | 1.777e-10 |                 1.066e-09 |                      1      |
| Guideline Alignment | Baseline vs Full pipeline        |                        1.78 | 5.964e-11 |                 5.368e-10 |                      1      |
| Guideline Alignment | Multi-step only vs Full pipeline |                        1.78 | 5.7e-11   |                 5.368e-10 |                      1      |
| Personalization     | RAG only vs Full pipeline        |                        0.12 | 0.01431   |                 0.04292   |                      1      |
| Safety              | Baseline vs Multi-step only      |                        0.2  | 0.003892  |                 0.01752   |                      0.8333 |
| Safety              | Multi-step only vs Full pipeline |                       -0.2  | 0.007526  |                 0.02709   |                     -0.7143 |

### Interpretation

- RAG produced very large and statistically significant gains in Guideline Alignment.
- Multi-step-only produced a statistically significant but smaller gain in Safety over Baseline.
- Full pipeline improved Guideline Alignment strongly, but did not outperform Multi-step-only on Safety.

## 3. Retrieval Evaluation

Retrieval performance was evaluated by checking whether the top-k retrieved chunks covered the expected guideline topic for each scenario, using normalized topic/keyword overlap.

| System        |   Recall@3 |   Recall@5 |   Mean Coverage@3 |   Mean Coverage@5 |   N |
|:--------------|-----------:|-----------:|------------------:|------------------:|----:|
| Full pipeline |      1.000 |      1.000 |             0.938 |             0.938 |  50 |
| RAG only      |      1.000 |      1.000 |             0.938 |             0.938 |  50 |

### Interpretation

Both RAG-only and Full pipeline achieved **Recall@3 = 1.00** and **Recall@5 = 1.00**, supporting that the RAG module consistently retrieved relevant guideline chunks. This strengthens the claim that RAG's higher guideline-alignment scores are supported by actual retrieval behavior.

## 4. Judge Self-Consistency

We re-evaluated a random subset of outputs to assess the stability of the LLM-as-judge scoring.

| Metric | Value |
|---|---:|
| Re-scored pairs | 80 |
| Exact agreement rate | 0.975 |
| Average absolute difference | 0.025 |
| Spearman correlation | 0.961 |
| Weighted Cohen's kappa | 0.964 |

### Interpretation

The LLM judge showed strong self-consistency. This supports using LLM-as-judge scores as comparative signals, while still treating them as imperfect evaluations.

## 5. Human Validation

We conducted blind human validation on 40 sampled outputs scored by two human raters. Raters did not see which system generated each output.

| Validation Metric | Value |
|---|---:|
| Number of outputs | 40 |
| Number of raters | 2 |
| Human-human exact agreement | 0.638 |
| Human-human avg. absolute difference | 0.462 |
| Human-human Spearman | 0.414 |
| Human-human weighted kappa | 0.407 |
| Human-LLM rounded exact agreement | 0.562 |
| Human-LLM avg. absolute difference | 0.519 |
| Human-LLM Spearman | 0.292 |
| Human-LLM weighted kappa | 0.225 |

### Interpretation

Human validation supports the LLM-as-judge evaluation but does not show perfect agreement. Human-human agreement was moderate, and human-vs-LLM agreement was reasonable but imperfect. Therefore, LLM judge scores should be interpreted as useful comparative signals rather than absolute ground truth.

## 6. Scenario-Type Breakdown

We also grouped scores by scenario type. Across all scenario categories, RAG-based systems consistently achieved higher Guideline Alignment than non-RAG systems. This suggests the RAG effect is not limited to a single category.

Key qualitative pattern:

- RAG improves guideline alignment across healthy adult, hypertension, obesity, prediabetes/diabetes risk, injury/limited mobility, and pregnancy/special caution scenarios.
- Multi-step-only safety gains are more scenario-dependent.
- Personalization differences remain modest across categories.

Detailed table: `results/scenario_type_breakdown_wide.csv`

## 7. Output Length Check

We checked output length to assess possible LLM-judge verbosity bias.

| System          |   Mean Word Count |   Median Word Count |   Mean Sentence Count |   N |
|:----------------|------------------:|--------------------:|----------------------:|----:|
| Baseline        |            296.48 |              297.00 |                 26.42 |  50 |
| Full pipeline   |            382.02 |              385.00 |                 30.38 |  50 |
| Multi-step only |            407.16 |              407.50 |                 33.28 |  50 |
| RAG only        |            226.80 |              226.00 |                 15.82 |  50 |

### Interpretation

Multi-step-only and Full pipeline outputs were longer on average. This means small gains in personalization or safety may partly reflect greater detail. However, RAG-only outputs were the shortest on average while still achieving much higher Guideline Alignment, so the main RAG result is unlikely to be explained by output length alone.

## 8. Counterfactual Personalization Case Study

| Counterfactual Type      |   Count |   Mean Score |
|:-------------------------|--------:|-------------:|
| meaningful_health_change |       5 |         4.60 |
| negative_control         |       5 |         5.00 |

### Interpretation

The Full pipeline responded appropriately to meaningful health-profile changes and remained relatively stable under irrelevant negative-control changes. This supports personalization qualitatively, even though average personalization score differences were modest.

## 9. Qualitative Error Taxonomy

Main qualitative findings:

| Category | Interpretation |
|---|---|
| Generic advice | Some outputs are safe and relevant but not deeply personalized. |
| Unsupported guideline-like claims | Non-RAG systems often give plausible advice without explicit evidence grounding. |
| Missing caution | Baseline/RAG-only outputs can under-emphasize caution for high-risk profiles. |
| Over-conservatism | Multi-step-only can be safer but sometimes overly cautious. |
| Specificity-safety trade-off | Full pipeline is more grounded and specific, but Multi-step-only is sometimes safer. |
| Limited personalization separation | Basic prompts already use structured profile fields, making score differences small. |

Detailed notes: `results/case_studies_notes.csv`

## 10. Overall Conclusion

The main project finding is that **RAG substantially improves guideline alignment**, while **multi-step safety mechanisms improve conservative risk handling**. The Full pipeline provides the best overall balance but does not dominate every metric. Evaluation checks, including retrieval evaluation, judge self-consistency, blind human validation, scenario-type breakdown, output-length analysis, and counterfactual case studies, support the reliability and interpretability of the results.
