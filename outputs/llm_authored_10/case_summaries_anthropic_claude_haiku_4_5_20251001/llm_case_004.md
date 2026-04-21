# llm_case_004 (medium)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_004/source_spec.json`
- description: `llm_case_004/description_direct/description.json`
- structured_ir: `llm_case_004/structured_intermediate/intermediate.json`

## Description
- auto_score: 79.72%
- strict_material_f1: 57.14%
- coarse_material_f1: 80.00%
- dimension_score: 90.30%

## Rebuild Comparison
- direct IoU/F1/material/correct: 36.09% / 53.04% / 33.05% / 13.52%
- structured IoU/F1/material/correct: 28.02% / 43.77% / 46.19% / 14.28%

## Repair Effort
- direct normalized_edit_distance: 1.8392
- structured normalized_edit_distance: 2.2114
- direct edits(add/del/rep): 581 overlap ref, 190/839/389
- structured edits(add/del/rep): 578 overlap ref, 210/1309/186

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
