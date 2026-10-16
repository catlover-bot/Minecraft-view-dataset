# New exploratory CPU attribute fidelity associations

This is a new exploratory CPU-derived attribute fidelity proxy, not conventional description quality. Historical RQ2 remains **NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES**. All 180 frozen O JSON records matched their original source bytes, input IDs and GT; no historical quality scores were substituted. The five fixed features and all 360 planned analyses are retained. In the score JSON, the inherited `score_status=NOT_COMPUTED_BY_SOURCE_AUDIT` describes Phase 0 only; the new `scores` object and `scoring_specification_id` identify the completed Phase 1 computation.

Source association JSON SHA256: `92ce7356a4313b85e6bcd4adb87abed0d48ecfbdf51320e16aa569ec8afe5829`. Analysis code: `6a5b2b99f4566bbd7dde5dbc7c44c866b19fe802`. See [definition](definition.md), [statistics specification](stats_definition.md), and [all 360 records](phase1/exploratory_associations_360.csv).

The composite uses weights 0.30/0.25/0.25/0.20 for exact material-set F1, coarse material-set F1, dimension relative accuracy and typed field completeness. Missing dimensions never borrow GT. Compared with the legacy script, strict IDs remain exact and unknown OTHER matches are restricted. All dimensions in these 180 records were valid; 10 descriptions contained at least one ambiguous/invalid material token and received the predefined unmatched-token treatment. Completeness was 1 for 178 records and 0.2 for 2, producing sparse variation.

## Pooled description-level Spearman, valid-only O IoU

Intervals are 95% building-cluster percentile intervals with 2000 draws, seed 20261016. The two Builder outcomes are repeated observations, not 360 independent descriptions. All rows below have 60 contributing buildings. OpenAI has 177 valid description pairs, Anthropic 179, and the both-valid arithmetic mean 176. Each component is reported separately; this table does not choose a best-performing component.

| Feature | OpenAI (177) | Anthropic (179) | Both-valid mean (176) |
|---|---:|---:|---:|
| strict material F1 | 0.1842 [0.0361, 0.3412] | 0.2058 [0.0459, 0.3670] | 0.2144 [0.0511, 0.3747] |
| coarse material F1 | 0.1271 [-0.0273, 0.2816] | 0.1147 [-0.0383, 0.2675] | 0.1406 [-0.0220, 0.2980] |
| dimension accuracy | 0.5697 [0.4593, 0.6599] | 0.5453 [0.4298, 0.6398] | 0.5943 [0.4894, 0.6823] |
| completeness | -0.0272 [CI unavailable] | 0.0802 [CI unavailable] | 0.0317 [CI unavailable] |
| composite | 0.3932 [0.2514, 0.5217] | 0.3745 [0.2278, 0.5126] | 0.4183 [0.2662, 0.5488] |

## Within-building composite associations

Centering uses complete pairs within each building. The rank association is Pearson correlation of within-building midrank deviations, without another global rank transformation. It addresses additive building offsets but does not establish causality or remove shared-GT dependence.

| Target | Association | Pairs / buildings | Coefficient [95% CI] |
|---|---|---:|---:|
| openai | within_building_centered_pearson | 177 / 60 | 0.4033 [0.2204, 0.6077] |
| openai | within_building_midrank_association | 177 / 60 | 0.3939 [0.2294, 0.5528] |
| anthropic | within_building_centered_pearson | 179 / 60 | 0.4042 [0.2409, 0.5761] |
| anthropic | within_building_midrank_association | 179 / 60 | 0.3502 [0.1667, 0.5190] |
| both_builders_mean | within_building_centered_pearson | 176 / 60 | 0.4478 [0.2607, 0.6358] |
| both_builders_mean | within_building_midrank_association | 176 / 60 | 0.4035 [0.2444, 0.5612] |

## Description-provider strata, composite pooled Spearman

The description provider generated the saved visual-description record; the Builder generated a reconstruction. Within a single description-provider stratum, one description per building makes within-building analysis nonestimable. All 180 such planned rows are retained explicitly.

| Description provider | Builder target | Pairs / buildings | Spearman [95% CI] |
|---|---|---:|---:|
| openai | openai | 57 / 57 | 0.0892 [-0.1957, 0.3685] |
| openai | anthropic | 59 / 59 | 0.1382 [-0.1401, 0.4023] |
| openai | both_builders_mean | 56 / 56 | 0.1145 [-0.1670, 0.3926] |
| anthropic | openai | 60 / 60 | 0.3553 [0.0974, 0.5792] |
| anthropic | anthropic | 60 / 60 | 0.3353 [0.0730, 0.5647] |
| anthropic | both_builders_mean | 60 / 60 | 0.3687 [0.1136, 0.6036] |
| gemini | openai | 60 / 60 | 0.1913 [-0.0673, 0.4299] |
| gemini | anthropic | 60 / 60 | 0.0714 [-0.1876, 0.3156] |
| gemini | both_builders_mean | 60 / 60 | 0.1449 [-0.1147, 0.3905] |

## Bounded-IoU failure0 sensitivity, separate from valid-only

Only the 3 OpenAI and 1 Anthropic identified O validation failures receive 0. No operational unknown is imputed. Global 942 is a J observation and remains null in the unchanged 960-slot source. The selected successful O retry at global 940 carries its original 401 provenance. Each target below uses 180 descriptions/60 buildings; the mean requires both eligible Builder outcomes.

| Target | Composite Spearman [95% CI] |
|---|---:|
| openai | 0.4188 [0.2797, 0.5462] |
| anthropic | 0.3736 [0.2271, 0.5104] |
| both_builders_mean | 0.4328 [0.2846, 0.5585] |

## Nonestimability and interpretation

Across 360 planned records:144 have point estimates and cluster CIs;24 have point estimates but insufficient finite bootstrap replicates for the prespecified 95% interval rule;12 have constant proxy;180 have one description per building and no within-building estimand. No unavailable coefficient/interval is replaced with 0. The 24 unstable completeness analyses retain their point estimate and report the bootstrap validity count. All features, strata and failure0 companions remain visible.

The composite has a positive descriptive association with O IoU in this frozen cohort, including the reported within-building summaries. This is not evidence that conventional caption quality predicts reconstruction. GT-derived attributes and reconstruction scores share GT; the feature definitions and arbitrary fixed weights also constrain interpretation. Multiple exploratory associations, source selection, ties, sparse completeness variation, fixed models/prompts, single generation calls and known output failures limit generalization. No exploratory p-values, significance labels, optimized weights, or additions to the original 20 Holm comparisons were produced.

An API-free next step is to inspect source-bound error examples and explain which attributes are represented or missing, while keeping these proxy results separate from conventional RQ2. A new quality measure would require its own fixed definition and validity argument.
