# llm_case_009 (complex)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_009/source_spec.json`
- description: `llm_case_009/description_direct/description.json`
- structured_ir: `llm_case_009/structured_intermediate/intermediate.json`

## Description
- auto_score: 68.88%
- strict_material_f1: 54.55%
- coarse_material_f1: 85.71%
- dimension_score: 44.34%

## Rebuild Comparison
- direct IoU/F1/material/correct: 10.43% / 18.89% / 0.00% / 0.00%
- structured IoU/F1/material/correct: 10.32% / 18.72% / 47.97% / 27.57%

## Repair Effort
- direct normalized_edit_distance: 1.0985
- structured normalized_edit_distance: 1.0676
- direct edits(add/del/rep): 367 overlap ref, 3041/325/260
- structured edits(add/del/rep): 369 overlap ref, 2967/308/249

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
