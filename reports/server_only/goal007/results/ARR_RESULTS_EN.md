# Goal007 ARR Results / Discussion / Limitations draft

Analysis label: **fixed60 analysis with one operational missing observation**.

## Results and RQ1: reconstruction from saved image descriptions

The frozen design contains 960 slots across 60 buildings, two Builders, and three saved description providers. O and J contain 360 slots each; P and E contain 120 each. Oracle conditions are not replicated across description providers. The selected observations comprise 959 identified completed outcomes, including 25 model-output failures, and 1 operationally unmeasurable slot. There are 934 valid aligned-IoU observations and 959 observations eligible for the auxiliary bounded-IoU analysis. These counts are not interchangeable with 961 fixed-scope attempts and 991 historical calls. This analysis sent zero research requests.

| Builder | Condition | IoU (valid slots/buildings) | MAE (valid slots/buildings) | F0 IoU (eligible slots) |
|---|---|---:|---:|---:|
| OpenAI | O | 0.1760 (177/60) | 12.8259 (177/60) | 0.1735 (180) |
| OpenAI | J | 0.1712 (177/60) | 13.1639 (177/60) | 0.1696 (179) |
| OpenAI | P | 0.1674 (60/60) | 4.4000 (60/60) | 0.1674 (60) |
| OpenAI | E | 0.9219 (55/55) | 0.3758 (55/55) | 0.8451 (60) |
| Anthropic | O | 0.1772 (179/60) | 13.3870 (179/60) | 0.1765 (180) |
| Anthropic | J | 0.1754 (180/60) | 13.5648 (180/60) | 0.1754 (180) |
| Anthropic | P | 0.1691 (60/60) | 0.0444 (60/60) | 0.1691 (60) |
| Anthropic | E | 0.9919 (46/46) | 0.0362 (46/46) | 0.7604 (60) |

All means above weight buildings equally after averaging eligible observations within each building. IoU is on [0,1], with larger values better; dimension MAE is in blocks, with smaller values better. Secondary geometry, material-aware IoU, material-set F1, repair cost, and addition/deletion/replacement counts are available with metric-specific denominators in `per_group.csv`. The saved description provider and the Builder are different experimental roles; all twelve description-provider × Builder × O/J combinations are included in `per_group.csv` and `figure_description_builders`.

Global940 uses the explicitly approved manual authentication retry after the original HTTP401. The original attempt and its unknown-usage reservation remain in history. Global942 (building_085, J, OpenAI Builder) had HTTP520 with unknown usage and remains `operational_unmeasurable`: its response, usage, and metrics are null. It is not a completed quality observation or a model-output failure and contributes no zero to any primary or failure0 analysis. Both unknown-usage holds remain in the separate accounting audit; calculated usage cost is not an invoice.

## RQ3 and the predeclared paired comparisons

J is a lossless JSON reserialization of the original O description, not the historical Structured IR intervention. P provides partial GT attributes and E provides exact discrete GT. Oracle contrasts include Builder capability and representation/execution effects; they do not identify a pure causal effect of language-information loss.

| Family | Contrast | Metric | Pairs/buildings | Delta [95% CI] | Raw p | Holm p |
|---|---|---|---:|---:|---:|---:|
| primary_builder_valid_pair | Anthropic - OpenAI / O | IoU | 176/60 | 0.0006 [-0.0056, 0.0069] | 0.860714 | 1 |
| primary_builder_valid_pair | Anthropic - OpenAI / O | MAE | 176/60 | 0.6259 [0.3963, 0.8584] | 9.999e-05 | 0.00079992 |
| primary_builder_valid_pair | Anthropic - OpenAI / J | IoU | 177/60 | 0.0043 [-0.0041, 0.0126] | 0.336866 | 1 |
| primary_builder_valid_pair | Anthropic - OpenAI / J | MAE | 177/60 | 0.4185 [0.0463, 0.8333] | 0.0343966 | 0.206379 |
| primary_builder_valid_pair | Anthropic - OpenAI / P | IoU | 60/60 | 0.0017 [-0.0159, 0.0184] | 0.845115 | 1 |
| primary_builder_valid_pair | Anthropic - OpenAI / P | MAE | 60/60 | -4.3556 [-5.4500, -3.3833] | 9.999e-05 | 0.00079992 |
| primary_builder_valid_pair | Anthropic - OpenAI / E | IoU | 46/46 | -0.0028 [-0.0105, 0.0062] | 0.526247 | 1 |
| primary_builder_valid_pair | Anthropic - OpenAI / E | MAE | 46/46 | 0.0362 [0.0072, 0.0652] | 0.0653935 | 0.326967 |
| secondary_lossless_J_minus_O | J - O / OpenAI | IoU | 175/60 | -0.0057 [-0.0122, 0.0003] | 0.0768923 | 0.230677 |
| secondary_lossless_J_minus_O | J - O / OpenAI | MAE | 175/60 | 0.3352 [0.1138, 0.5630] | 0.00319968 | 0.0127987 |
| secondary_lossless_J_minus_O | J - O / Anthropic | IoU | 179/60 | -0.0016 [-0.0092, 0.0056] | 0.690331 | 0.966703 |
| secondary_lossless_J_minus_O | J - O / Anthropic | MAE | 179/60 | 0.1157 [-0.1676, 0.4519] | 0.483352 | 0.966703 |
| diagnostic_oracle_minus_original | P - O / OpenAI | IoU | 60/60 | -0.0087 [-0.0248, 0.0066] | 0.289771 | 0.493951 |
| diagnostic_oracle_minus_original | P - O / OpenAI | MAE | 60/60 | -8.4259 [-9.5463, -7.2100] | 9.999e-05 | 0.00079992 |
| diagnostic_oracle_minus_original | E - O / OpenAI | IoU | 55/55 | 0.7458 [0.6862, 0.8017] | 9.999e-05 | 0.00079992 |
| diagnostic_oracle_minus_original | E - O / OpenAI | MAE | 55/55 | -12.3697 [-13.2033, -11.5323] | 9.999e-05 | 0.00079992 |
| diagnostic_oracle_minus_original | P - O / Anthropic | IoU | 60/60 | -0.0081 [-0.0207, 0.0062] | 0.246975 | 0.493951 |
| diagnostic_oracle_minus_original | P - O / Anthropic | MAE | 60/60 | -13.3426 [-14.0315, -12.7000] | 9.999e-05 | 0.00079992 |
| diagnostic_oracle_minus_original | E - O / Anthropic | IoU | 46/46 | 0.8137 [0.8007, 0.8262] | 9.999e-05 | 0.00079992 |
| diagnostic_oracle_minus_original | E - O / Anthropic | MAE | 46/46 | -13.4010 [-14.2368, -12.5458] | 9.999e-05 | 0.00079992 |

The three families contain 8, 4, and 8 comparisons. At Holm-adjusted p < 0.05, respectively 2, 1, and 6 comparisons pass that threshold; the table retains every contrast regardless of significance. Effects are B−A on common-valid pairs, averaged within buildings and then equally across buildings. Positive IoU differences favor B; positive MAE differences indicate greater error for B. Statistical non-significance alone does not establish equivalence. Auxiliary failure0 effects and eligible pair/building counts are in `comparisons_20.csv` and the LaTeX tables; they are bounded-IoU descriptive companions, not additional significance tests.

## RQ2: description-quality scores versus reconstruction

**NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES**. Existing historical per-sample scores were identified, but their source provenance does not establish strict correspondence to all current frozen v4 descriptions. Some historical description fields differ and Gemini matched scores are absent. Historical model means, old v5 records, and three-model-average correlations are not substituted. No new RQ2 correlation or bootstrap is reported. See `rq2_matched_scores_audit.json` for searched paths, content hashes, and exclusion evidence.

## Missing-observation sensitivity and limitations

The primary analysis excludes operationally missing measurements from common-valid pairs while retaining the slot in the design. `sensitivity_missing942.json` separately records affected contrasts and exploratory aligned-IoU endpoint bounds using hypothetical values 0 and 1. These endpoints are possible-value assumptions, not imputations, observed failures, confidence intervals, or new p-values; no post-hoc significance conclusion is drawn. Unbounded MAE has no analogous finite [0,1] bound.

The independent sample is at most 60 buildings, not 960 independent trials. Selection on valid output can alter the main estimand; bounded-IoU companions show the effect of known output failures but cannot resolve unknown operational outcomes. The experiment fixes models, prompts, saved descriptions, GT, and evaluator implementation. It supports claims about this measured voxel reconstruction setting; broader scenes, human utility, language understanding, and model populations require separate evidence. Equal-building weights prevent description-provider repetitions from inflating independent N. Operational missingness, selected approved retry provenance, model-output failures, unavailable strictly matched quality scores, and oracle interpretation limit any claim that language descriptions generally preserve sufficient information.

CPU-only priorities are to inspect the existing metric/failure breakdowns and paired interval widths, review existing saved descriptions and reconstruction examples without altering observations, and, if a new scoring protocol is later specified, construct provenance-bound description-quality evidence as a separately labeled exploratory analysis. No additional generation is needed for those audits.
