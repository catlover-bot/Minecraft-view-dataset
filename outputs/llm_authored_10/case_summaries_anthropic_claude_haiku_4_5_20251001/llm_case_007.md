# llm_case_007 (medium)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_007/source_spec.json`
- description: `llm_case_007/description_direct/description.json`
- structured_ir: `llm_case_007/structured_intermediate/intermediate.json`

## Description
- auto_score: 58.21%
- strict_material_f1: 28.57%
- coarse_material_f1: 40.00%
- dimension_score: 78.57%

## Rebuild Comparison
- direct IoU/F1/material/correct: 22.58% / 36.84% / 35.29% / 14.05%
- structured IoU/F1/material/correct: 20.15% / 33.54% / 32.04% / 10.60%

## Repair Effort
- direct normalized_edit_distance: 1.3971
- structured normalized_edit_distance: 1.5785
- direct edits(add/del/rep): 884 overlap ref, 1695/1336/572
- structured edits(add/del/rep): 877 overlap ref, 1702/1773/596

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
