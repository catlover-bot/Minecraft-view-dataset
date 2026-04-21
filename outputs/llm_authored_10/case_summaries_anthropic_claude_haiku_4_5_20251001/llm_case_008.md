# llm_case_008 (complex)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_008/source_spec.json`
- description: `llm_case_008/description_direct/description.json`
- structured_ir: `llm_case_008/structured_intermediate/intermediate.json`

## Description
- auto_score: 63.16%
- strict_material_f1: 44.44%
- coarse_material_f1: 57.14%
- dimension_score: 62.16%

## Rebuild Comparison
- direct IoU/F1/material/correct: 19.28% / 32.32% / 29.72% / 6.83%
- structured IoU/F1/material/correct: 18.03% / 30.56% / 25.74% / 5.70%

## Repair Effort
- direct normalized_edit_distance: 2.6657
- structured normalized_edit_distance: 2.6620
- direct edits(add/del/rep): 1753 overlap ref, 1463/5878/1232
- structured edits(add/del/rep): 1585 overlap ref, 1778/5720/1063

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
