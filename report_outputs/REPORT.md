# Bias Explainability Report (Bias in Bios)

## Question answered
How do attention-based explanations and post-hoc attribution methods (SHAP/LIME) differ for revealing and interpreting biased outputs, and how well do they align with human intuition?

## Model + bias stress-test snapshot
- Overall accuracy: **0.667**
- Gender-0 accuracy: **0.651**
- Gender-1 accuracy: **0.686**
- Counterfactual prediction-change rate after gender swapping: **0.137**

## Interpretation
- Attention provides intrinsic, model-internal token relevance but is often diffuse.
- LIME gives sparse local feature weights that are easy to inspect case-by-case.
- SHAP provides additive and signed attributions with stronger cross-case consistency (but higher compute).
- Alignment with human intuition is approximated with top-k gender token hit rate and attribution mass on gendered words in counterfactual flips.

## Method ranking (from this run)
- SHAP: top-k hit=0.700, gender-mass=0.167, mrr=0.401
- LIME: top-k hit=0.600, gender-mass=0.173, mrr=0.402
- Attention: top-k hit=0.200, gender-mass=0.110, mrr=0.284

## Artifacts
- `table_01_model_metrics.csv`
- `table_02_profession_gap_top10.csv`
- `table_03_counterfactual_examples.csv`
- `table_04_method_alignment_summary.csv`
- `figure_01_swap_change_rate.png`
- `figure_02_topk_hit_rate.png`
- `figure_03_gender_mass_boxplot.png`