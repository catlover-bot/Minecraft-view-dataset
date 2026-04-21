# llm_case_001 (simple)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_001/source_spec.json`
- description: `llm_case_001/description_direct/description.json`
- structured_ir: `llm_case_001/structured_intermediate/intermediate.json`

## Description
- auto_score: 61.57%
- strict_material_f1: 28.57%
- coarse_material_f1: 50.00%
- dimension_score: 82.01%

## Rebuild Comparison
- direct IoU/F1/material/correct: 34.12% / 50.88% / 0.00% / 0.00%
- structured IoU/F1/material/correct: 39.57% / 56.71% / 33.85% / 14.52%

## Repair Effort
- direct normalized_edit_distance: 1.9035
- structured normalized_edit_distance: 1.8296
- direct edits(add/del/rep): 202 overlap ref, 109/281/202
- structured edits(add/del/rep): 260 overlap ref, 51/346/172

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
