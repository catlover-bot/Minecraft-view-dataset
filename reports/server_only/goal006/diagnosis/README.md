# Goal006 saved-result diagnosis

This analysis follows the verified GitHub preservation of all 160 completed batch outcomes. It uses the saved plan, raw responses, parsed operations, voxel artifacts, and evaluation dictionaries. Additional research API sends: **0**. The 641 original files listed in `../results/verification.json` retained their hashes. The prior audit's 154 raw→parsed→executed-voxel checks are reused; no successful executor chain was repeated here.

Eight representative metrics were fixed as the first successful scheduled row for each provider×condition, without consulting score values. Each full metric dictionary was recomputed exactly once and matched the saved object, including alignment metadata. All eight are building 087, so this is a representative semantic check rather than a re-evaluation of every size stratum. Diagnosis took 5.376 seconds. Eleven new network-blocked fixture tests passed. No measurement mismatch was found in these checks; no metric version, prompt, input, model, cap, validator, or raw outcome changed.

The six failures are preserved as failures. Operation numbers below are one-based; JSON evidence also records zero-based indices, attempt IDs, raw SHA256, the actual operation, and frozen rule.

| Provider | Building / condition / description source | Actual failure evidence |
|---|---|---|
| OpenAI | 078 / J / OpenAI | Operation 4, `block="wood (plank, oak)"` |
| OpenAI | 062 / J / OpenAI | Operation 2, `block="quartz (white)"` |
| OpenAI | 062 / O / OpenAI | Operation 2, `block="sandstone (or smooth_sand)"` |
| Anthropic | 032 / E | `stop_reason=max_tokens`, output 32768 |
| OpenAI | 092 / E | `status=incomplete`, `reason=max_output_tokens`, output 32768 including reasoning 384 |
| Anthropic | 092 / E | `stop_reason=max_tokens`, output 32768 |

All three validation errors pass the frozen shape schema, then violate the exact lower-case material identifier regex `(?:minecraft:)?[a-z0-9_][a-z0-9_./-]{0,127}\Z`. No automatic normalization or GT mapping was applied. The local reference-action token counts are 51111 for 032 and 34574 for 092; both exceed 32768. This is a mechanical association, not proof that the reference is minimal or that every response must truncate. Anthropic's exact reference token count is unknown. Actual provider finish reasons establish the three truncations. See [failures.json](failures.json).

P supplies four GT-derived attributes only: dimensions, world bounding box, material set, and voxel count. It contains neither the original description nor per-coordinate geometry/materials. The diagnostic compares every one of the 20 supplied attribute records with its saved prediction, retaining signed errors (`predicted − input`), material presence errors, and separate satisfaction flags.

| Provider | Exact dimensions | Exact bbox | Exact material set | Exact voxel count | All four exact |
|---|---:|---:|---:|---:|---:|
| OpenAI | 1/10 | 1/10 | 6/10 | 4/10 | 0/10 |
| Anthropic | 10/10 | 10/10 | 8/10 | 0/10 | 0/10 |

Anthropic's voxel-count errors are all positive, ranging from +5101 to +39841. Its mean material-set F1 is 0.9874509804, aligned occupancy IoU 0.1867045685, and aligned material-aware IoU 0.0227370430. These results do not support saying that every supplied attribute was obeyed. For building 087, the input count is 3241 and the prediction count 13000 (+9759), although dimensions, bbox, and material set match. Saved operation 1 fills stone across x=[−3,28], y=[3,8], z=[−3,27]; final predicted coordinate (−3,3,−3) is stone while GT is air. This is direct evidence of a saved fill contributing extra occupied coordinates, without an inference about the model's hidden reasoning. Every P row includes bounded examples with the final covering operation, raw/parsed/voxel hashes, and saved metric dictionaries. See [p_attributes.json](p_attributes.json) and [p_attributes.csv](p_attributes.csv).

E is the complete discrete labelled GT, including interior and world coordinates. The table distinguishes accepted complete outputs, truncation treated as zero only for eligible bounded IoU, and the common-valid building intersection. Each mean weights buildings equally. These are interim descriptive results, without p-values or a provider superiority conclusion.

| Provider | Denominator | Aligned occupancy IoU | Aligned material-aware IoU |
|---|---|---:|---:|
| OpenAI | Valid only: 9 | 0.9411124005 | 0.7848044319 |
| OpenAI | All completed attempts, truncation=0: 10 | 0.8470011605 | 0.7063239887 |
| OpenAI | Common valid: 8 | 0.9807176367 | 0.8129041616 |
| Anthropic | Valid only: 8 | 0.9826017244 | 0.7815501833 |
| Anthropic | All completed attempts, truncation=0: 10 | 0.7860813795 | 0.6252401466 |
| Anthropic | Common valid: 8 | 0.9826017244 | 0.7815501833 |

The common-valid buildings are **007, 023, 026, 045, 055, 062, 078, 087**. Among valid outputs, OpenAI has 6/9 exact occupied-coordinate sets and 2/9 exact labelled-voxel sets; Anthropic has 2/8 and 1/8 respectively. Absolute and aligned exactness counts coincide here. Truncation is separate from a complete response that transcribes coordinates/materials incorrectly. Per-building exact-ID confusion and examples are in [e_diagnostics.json](e_diagnostics.json). E−O is not pure natural-language information loss.

J is information-preserving JSON reserialization of the saved O text; it is not the old ARR Structured IR. [denominators.json](denominators.json) separates valid-only means, eligible failure0 IoU means, common-valid paired J−O differences, building counts, description-source tables, and exact pair lists. O/J descriptions are averaged within each building before averaging buildings. OpenAI O and J have 29/30 and 28/30 valid rows, with 28 matched valid pairs over 10 buildings. Anthropic has 30/30 for each and 30 matched pairs over 10 buildings. For example, OpenAI's aligned occupancy IoU difference of separate valid group means is −0.0038757942, whereas its common-valid paired building difference is −0.0037366243. Those quantities are not interchangeable. Failed MAE and repair metrics remain null; unknown/not-run outcomes are never silently assigned failure0.

The CPU command takes a new output directory and the already verified completed-batch evidence:

```bash
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.goal006.diagnosis --plan /root/minecraft-runs/goal005-batch1-20261013T213420Z/plan.json --verified-evidence reports/server_only/goal006/results/verification.json --output NEW_UNIQUE_OUTPUT_DIRECTORY
```

The saved results above were computed once; this command documents reproducibility and is not an instruction to repeat the metric work during this Goal. No 154-output full metric rerun, 200-GT evaluation, or old 1200-output reevaluation was performed.
