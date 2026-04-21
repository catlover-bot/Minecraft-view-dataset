# llm_case_001 (simple)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_001/source_spec.json`
- description: `llm_case_001/description_direct/description.json`
- structured_ir: `llm_case_001/structured_intermediate/intermediate.json`

## Description
- auto_score: 69.61%
- strict_material_f1: 44.44%
- coarse_material_f1: 57.14%
- dimension_score: 87.96%

## Rebuild Comparison
- direct IoU/F1/material/correct: 22.90% / 37.26% / 0.00% / 0.00%
- structured IoU/F1/material/correct: 33.04% / 49.67% / 40.11% / 15.69%

## Repair Effort
- direct normalized_edit_distance: 3.2400
- structured normalized_edit_distance: 1.7855
- direct edits(add/del/rep): 204 overlap ref, 71/616/204
- structured edits(add/del/rep): 187 overlap ref, 88/291/112

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
