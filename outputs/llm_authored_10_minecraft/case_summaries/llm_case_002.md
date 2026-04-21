# llm_case_002 (simple)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_002/source_spec.json`
- description: `llm_case_002/description_direct/description.json`
- structured_ir: `llm_case_002/structured_intermediate/intermediate.json`

## Description
- auto_score: 81.33%
- strict_material_f1: 66.67%
- coarse_material_f1: 75.00%
- dimension_score: 90.30%

## Rebuild Comparison
- direct IoU/F1/material/correct: 36.88% / 53.89% / 0.34% / 0.15%
- structured IoU/F1/material/correct: 32.29% / 48.81% / 67.70% / 30.48%

## Repair Effort
- direct normalized_edit_distance: 1.9033
- structured normalized_edit_distance: 1.2901
- direct edits(add/del/rep): 298 overlap ref, 126/384/297
- structured edits(add/del/rep): 226 overlap ref, 198/276/73

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
