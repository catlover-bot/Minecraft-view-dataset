# llm_case_006 (medium)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_006/source_spec.json`
- description: `llm_case_006/description_direct/description.json`
- structured_ir: `llm_case_006/structured_intermediate/intermediate.json`

## Description
- auto_score: 68.84%
- strict_material_f1: 28.57%
- coarse_material_f1: 66.67%
- dimension_score: 94.41%

## Rebuild Comparison
- direct IoU/F1/material/correct: 29.26% / 45.28% / 11.72% / 5.50%
- structured IoU/F1/material/correct: 20.56% / 34.10% / 0.00% / 0.00%

## Repair Effort
- direct normalized_edit_distance: 1.4434
- structured normalized_edit_distance: 2.7587
- direct edits(add/del/rep): 290 overlap ref, 373/328/256
- structured edits(add/del/rep): 376 overlap ref, 287/1166/376

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
