# Interpretation supplement for the fixed60 analysis

Source: final `analysis-final/results/results.json`, SHA256 `166d46416fcbf36c450ffaf37310fc4f50ad48727fd6b4bb40d9767c38738dca`; final `analysis-final/MANIFEST.json`, SHA256 `ef05c2ed93247802e9a68a00d700d39a8934d6231c7787315ab37b01d2467714`; analysis code commit `7500cea2160a8489f0e8bd06b6249d261226f943`. This supplement interprets existing results and adds no test, comparison, score, or API request. The analysis label is **fixed60 analysis with one operational missing observation**.

The 960 fixed slots contain 934 valid reconstructions, 25 identified model-output failures (19 truncations and 6 validation failures), and one operationally unmeasurable HTTP520 observation. They represent 961 fixed-scope attempts; historical calls total 991. All means below average eligible descriptions within building and then weight buildings equally. Independent N never exceeds 60. Main comparisons use common-valid pairs; auxiliary failure0 applies only to known completed model-output failures in bounded IoU.

## RQ1: both geometry and material placement remain inaccurate under O

| Original O: valid-only mean | OpenAI Builder (177 slots/60 buildings) | Anthropic Builder (179 slots/60 buildings) |
|---|---:|---:|
| Absolute occupancy IoU | 0.076210 | 0.087371 |
| Strict-aligned occupancy IoU | 0.176032 | 0.177169 |
| Absolute material-aware IoU | 0.008791 | 0.008375 |
| Aligned material-aware IoU | 0.021005 | 0.016857 |
| Material-set F1 | 0.414532 | 0.445059 |
| Absolute normalized repair cost | 1.646643 | 2.024535 |
| Aligned normalized repair cost | 1.488769 | 1.873178 |

Allowing strict translation increases descriptive occupancy overlap, but aligned IoU remains below 0.18 in both O groups. Translation therefore removes some positional discrepancy while substantial geometric disagreement remains. Material-set overlap does not imply correct material placement: aligned material-aware IoU is only 0.021005 and 0.016857. This evaluator retains its exact material identities; no material synonym merging or GT-based prediction correction was introduced. These descriptive metrics do not establish a causal decomposition of the errors.

Repair cost is additions plus deletions plus replacements divided by GT occupied-voxel count for each input, followed by equal-building averaging; it can exceed 1. At the evaluator's aligned position, the mean addition/deletion/replacement counts are 3666.41/2622.64/1200.85 for OpenAI and 3364.82/4553.01/1511.94 for Anthropic. These are descriptive operation counts, not estimated human editing time or separately tested provider differences.

The predeclared O Builder contrast for aligned IoU is Anthropic−OpenAI +0.000593 (176 pairs/60 buildings; 95% CI [−0.005610,0.006937], Holm p=1). None of the four predeclared Builder-IoU contrasts passes Holm p<0.05. This does not establish equivalence, and larger material/repair differences above are not new significance claims.

## RQ3: accurate dimensions do not guarantee occupancy reconstruction

Under P, dimension MAE is 4.400000 for OpenAI and 0.044444 for Anthropic, while aligned occupancy IoU remains 0.167358 and 0.169093. The frozen P−O comparisons show lower dimension error but do not establish an IoU improvement. Thus partial GT attributes can support correct extents without resolving the internal occupied geometry in this setting. P is an oracle diagnostic containing extra GT information; it is not an estimate of pure language-information loss.

The following are selected **already predeclared** contrasts. All 20, including nonsignificant outcomes, remain in `table_statistics.tex` and `comparisons_20.csv`. Effects use the displayed B−A direction; negative MAE means less error. CIs are the frozen building-bootstrap 95% intervals and Holm p-values retain the original 8/4/8 family scopes.

| Predeclared contrast / metric | Common-valid pairs/buildings | Effect |95%CI| Holm p |
|---|---:|---:|---:|---:|
| P−O / OpenAI / MAE | 60/60 | -8.425926 | [-9.546319, -7.210046] | 0.00079992001 |
| P−O / Anthropic / MAE | 60/60 | -13.342593 | [-14.031528, -12.699954] | 0.00079992001 |
| P−O / OpenAI / IoU | 60/60 | -0.008675 | [-0.024784, 0.006609] | 0.4939506 |
| P−O / Anthropic / IoU | 60/60 | -0.008076 | [-0.020720, 0.006158] | 0.4939506 |
| E−O / OpenAI / IoU | 55/55 | +0.745755 | [0.686232, 0.801694] | 0.00079992001 |
| E−O / Anthropic / IoU | 46/46 | +0.813718 | [0.800704, 0.826220] | 0.00079992001 |
| Anthropic−OpenAI / E / IoU | 46/46 | -0.002797 | [-0.010450, 0.006185] | 1 |
| J−O / OpenAI / MAE | 175/60 | +0.335185 | [0.113843, 0.562963] | 0.01279872 |

E gives high valid-only aligned IoU: 0.921894 for OpenAI (55/60 valid) and 0.991872 for Anthropic (46/60 valid). However, E also has 5 and 14 model-output failures, respectively. Including those known failures as zero in the bounded-IoU companion gives 0.845069 and 0.760435 across 60 eligible inputs each. Failed MAE/repair remain null. Reporting only E's successful outputs would hide this operationally relevant model-output failure burden.

The unequal valid sets are crucial: the marginal E means 0.921894 versus 0.991872 do **not** establish Anthropic superiority. On the 46 common-valid buildings, the paired Anthropic−OpenAI IoU effect is −0.002797 with Holm p=1. E demonstrates that the Builders can often reconstruct exact discrete specifications, conditional on producing valid output; it also exposes output/execution constraints. It does not isolate the causal amount of information lost in the O descriptions.

J is lossless JSON reserialization of the same saved O information, not the historical Structured IR intervention. OpenAI J−O dimension MAE is higher by 0.335185 blocks (table above), but its aligned-IoU effect is −0.005697 (175 pairs/60 buildings; 95% CI [−0.012164,0.000300], Holm p=0.23067693). Anthropic J−O IoU is −0.001581 (179 pairs/60 buildings; 95% CI [−0.009151,0.005587], Holm p=0.96670333). These results do not support a general reconstruction benefit from this lossless reserialization; they do not test a richer attribute-IR intervention.

## Missingness, RQ2, and the boundary of the claim

Global940 uses its approved manual authentication retry success while retaining the original 401 attempt and unknown-usage hold. Global942 remains an unmeasured HTTP520 slot, never an observed IoU 0 or model-output failure. Both unknown-usage holds of USD 0.165536 remain. The accounting total USD 21.56083625 combines known-usage calculations and conservative reservations; it is not an invoice.

Separate exploratory endpoint bounds assume the unmeasured IoU could lie anywhere in [0,1]. For J's Anthropic−OpenAI comparison, the hypothetical common-valid mean difference spans [0.000247902,0.005803458] with 178 pairs/60 buildings, versus 177 pairs in the observed main analysis. For OpenAI J−O it spans [−0.006564128,−0.001008573] with 176 pairs/60 buildings, versus 175 observed pairs. Its auxiliary failure0-companion bound spans [−0.005552970,0.000002586]. These are conditional possible-value bounds, not confidence intervals, observed outcomes, or new significance tests. No finite [0,1] bound is assigned to unbounded MAE.

RQ2 is **NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES**. Historical per-sample scores exist, but they lack the necessary source-content binding; 40 of 60 matched-building Claude records disagree with current descriptions on dimensions, and matched Gemini scores are absent. OpenAI dimension agreement alone cannot prove full-description identity. Neither old model averages nor v5 data were joined to the current results.

The measured pipeline leaves substantial reconstruction error under saved image-derived descriptions, but these 60 buildings, two fixed Builders, fixed prompts and evaluator cannot by themselves establish universal language insufficiency. Valid-only selection, model-output failures, one operational missing observation, and absent strictly matched quality scores constrain that claim. A useful API-free next step is source-bound inspection of existing error examples and repair components, followed only under a separately specified protocol by new CPU-derived description-quality evidence labeled exploratory.
