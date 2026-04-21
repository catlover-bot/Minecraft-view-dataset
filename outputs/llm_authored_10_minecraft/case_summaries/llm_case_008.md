# llm_case_008 (complex)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_008/source_spec.json`
- description: `llm_case_008/description_direct/description.json`
- structured_ir: `llm_case_008/structured_intermediate/intermediate.json`

## Description
- auto_score: 78.48%
- strict_material_f1: 60.00%
- coarse_material_f1: 85.71%
- dimension_score: 76.19%

## Rebuild Comparison
- direct IoU/F1/material/correct: 29.87% / 46.01% / 8.89% / 6.20%
- structured IoU/F1/material/correct: 25.19% / 40.24% / 38.19% / 21.61%

## Repair Effort
- direct normalized_edit_distance: 1.1187
- structured normalized_edit_distance: 1.1204
- direct edits(add/del/rep): 596 overlap ref, 1140/259/543
- structured edits(add/del/rep): 542 overlap ref, 1194/416/335

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
