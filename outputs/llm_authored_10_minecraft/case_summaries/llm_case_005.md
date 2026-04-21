# llm_case_005 (medium)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_005/source_spec.json`
- description: `llm_case_005/description_direct/description.json`
- structured_ir: `llm_case_005/structured_intermediate/intermediate.json`

## Description
- auto_score: 77.40%
- strict_material_f1: 50.00%
- coarse_material_f1: 85.71%
- dimension_score: 83.87%

## Rebuild Comparison
- direct IoU/F1/material/correct: 32.44% / 48.99% / 51.47% / 29.22%
- structured IoU/F1/material/correct: 42.35% / 59.50% / 63.88% / 41.96%

## Repair Effort
- direct normalized_edit_distance: 1.1062
- structured normalized_edit_distance: 0.9368
- direct edits(add/del/rep): 511 overlap ref, 675/389/248
- structured edits(add/del/rep): 645 overlap ref, 541/337/233

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
