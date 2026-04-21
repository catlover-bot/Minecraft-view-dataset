# llm_case_010 (complex)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_010/source_spec.json`
- description: `llm_case_010/description_direct/description.json`
- structured_ir: `llm_case_010/structured_intermediate/intermediate.json`

## Description
- auto_score: 61.35%
- strict_material_f1: 25.00%
- coarse_material_f1: 40.00%
- dimension_score: 95.38%

## Rebuild Comparison
- direct IoU/F1/material/correct: 20.60% / 34.16% / 0.00% / 0.00%
- structured IoU/F1/material/correct: 21.16% / 34.93% / 60.61% / 15.02%

## Repair Effort
- direct normalized_edit_distance: 3.8531
- structured normalized_edit_distance: 2.4344
- direct edits(add/del/rep): 1507 overlap ref, 392/5418/1507
- structured edits(add/del/rep): 1122 overlap ref, 777/3404/442

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
