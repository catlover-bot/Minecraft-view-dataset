# llm_case_004 (medium)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_004/source_spec.json`
- description: `llm_case_004/description_direct/description.json`
- structured_ir: `llm_case_004/structured_intermediate/intermediate.json`

## Description
- auto_score: 87.95%
- strict_material_f1: 88.89%
- coarse_material_f1: 100.00%
- dimension_score: 65.15%

## Rebuild Comparison
- direct IoU/F1/material/correct: 34.96% / 51.81% / 20.27% / 12.71%
- structured IoU/F1/material/correct: 47.10% / 64.04% / 58.02% / 44.65%

## Repair Effort
- direct normalized_edit_distance: 1.1730
- structured normalized_edit_distance: 0.8460
- direct edits(add/del/rep): 301 overlap ref, 381/179/240
- structured edits(add/del/rep): 374 overlap ref, 308/112/157

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
