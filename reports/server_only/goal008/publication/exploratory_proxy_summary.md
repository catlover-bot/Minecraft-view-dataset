# Exploratory CPU attribute proxy—not conventional description quality

A separate, new CPU-derived attribute fidelity proxy was specified before these scores and associations were calculated. It uses strict material F1, coarse material F1, missing-safe dimension accuracy and typed field completeness, with fixed weights 0.30/0.25/0.25/0.20. All 180 descriptions have exact original-source, input and GT hash binding. Missing or invalid dimensions are penalized rather than filled from GT. This proxy is not conventional description quality or a human judgment. All 360 specified associations are retained, including nonestimable strata and unstable intervals; no p-value or Holm family was added. Two Builder outcomes repeat each description. Intervals resample 60 building clusters 2,000 times with PCG64 seed 20261016; this seed is not a decoding seed.

OpenAI, valid_only, Pooled Spearman: coefficient 0.3932, 95% CI [0.2514, 0.5217], 177 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

OpenAI, valid_only, Within-building centered: coefficient 0.4033, 95% CI [0.2204, 0.6077], 177 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

OpenAI, valid_only, Within-building rank: coefficient 0.3939, 95% CI [0.2294, 0.5528], 177 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

Anthropic, valid_only, Pooled Spearman: coefficient 0.3745, 95% CI [0.2278, 0.5126], 179 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

Anthropic, valid_only, Within-building centered: coefficient 0.4042, 95% CI [0.2409, 0.5761], 179 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

Anthropic, valid_only, Within-building rank: coefficient 0.3502, 95% CI [0.1667, 0.5190], 179 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

Both-Builder mean, valid_only, Pooled Spearman: coefficient 0.4183, 95% CI [0.2662, 0.5488], 176 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

Both-Builder mean, valid_only, Within-building centered: coefficient 0.4478, 95% CI [0.2607, 0.6358], 176 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

Both-Builder mean, valid_only, Within-building rank: coefficient 0.4035, 95% CI [0.2444, 0.5612], 176 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

OpenAI, bounded_iou_failure0, Pooled Spearman: coefficient 0.4188, 95% CI [0.2797, 0.5462], 180 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

OpenAI, bounded_iou_failure0, Within-building centered: coefficient 0.3962, 95% CI [0.1800, 0.6262], 180 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

OpenAI, bounded_iou_failure0, Within-building rank: coefficient 0.4000, 95% CI [0.2415, 0.5583], 180 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

Anthropic, bounded_iou_failure0, Pooled Spearman: coefficient 0.3736, 95% CI [0.2271, 0.5104], 180 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

Anthropic, bounded_iou_failure0, Within-building centered: coefficient 0.4030, 95% CI [0.2409, 0.5751], 180 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

Anthropic, bounded_iou_failure0, Within-building rank: coefficient 0.3417, 95% CI [0.1667, 0.5083], 180 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

Both-Builder mean, bounded_iou_failure0, Pooled Spearman: coefficient 0.4328, 95% CI [0.2846, 0.5585], 180 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

Both-Builder mean, bounded_iou_failure0, Within-building centered: coefficient 0.4374, 95% CI [0.2431, 0.6407], 180 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

Both-Builder mean, bounded_iou_failure0, Within-building rank: coefficient 0.3917, 95% CI [0.2417, 0.5500], 180 pairs/60 buildings; ESTIMABLE (2000/2000 bootstrap estimates finite).

All 5 features, all 3 description-provider strata, all targets and both outcome policies appear in exploratory_appendix.tex and the unchanged 360-row CSV; their tie/missing/constant diagnostics remain in the source JSON. Status counts: {"CONSTANT_PROXY": 12, "ESTIMABLE": 144, "ESTIMATE_WITHOUT_CLUSTER_CI": 24, "NOT_ESTIMABLE_SINGLE_DESCRIPTION_PER_BUILDING": 180}.
