# llm_case_009 (complex)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_009/source_spec.json`
- description: `llm_case_009/description_direct/description.json`
- structured_ir: `llm_case_009/structured_intermediate/intermediate.json`

## Description
- auto_score: 71.71%
- strict_material_f1: 44.44%
- coarse_material_f1: 66.67%
- dimension_score: 86.85%

## Rebuild Comparison
- direct IoU/F1/material/correct: 37.98% / 55.05% / 3.19% / 1.42%
- structured IoU/F1/material/correct: 21.97% / 36.03% / 41.74% / 10.39%

## Repair Effort
- direct normalized_edit_distance: 1.8787
- structured normalized_edit_distance: 2.6972
- direct edits(add/del/rep): 1066 overlap ref, 410/1331/1032
- structured edits(add/del/rep): 963 overlap ref, 513/2907/561

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
