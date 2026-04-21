# llm_case_003 (simple)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_003/source_spec.json`
- description: `llm_case_003/description_direct/description.json`
- structured_ir: `llm_case_003/structured_intermediate/intermediate.json`

## Description
- auto_score: 69.44%
- strict_material_f1: 50.00%
- coarse_material_f1: 66.67%
- dimension_score: 71.11%

## Rebuild Comparison
- direct IoU/F1/material/correct: 46.43% / 63.42% / 32.96% / 15.83%
- structured IoU/F1/material/correct: 44.10% / 61.21% / 31.14% / 14.03%

## Repair Effort
- direct normalized_edit_distance: 1.7028
- structured normalized_edit_distance: 1.8671
- direct edits(add/del/rep): 267 overlap ref, 19/289/179
- structured edits(add/del/rep): 273 overlap ref, 13/333/188

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
