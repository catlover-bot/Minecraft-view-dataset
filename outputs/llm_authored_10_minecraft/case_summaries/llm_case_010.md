# llm_case_010 (complex)

- provider_tag: `anthropic_claude_haiku_4_5_20251001`
- source_spec: `llm_case_010/source_spec.json`
- description: `llm_case_010/description_direct/description.json`
- structured_ir: `llm_case_010/structured_intermediate/intermediate.json`

## Description
- auto_score: 87.38%
- strict_material_f1: 90.91%
- coarse_material_f1: 100.00%
- dimension_score: 60.45%

## Rebuild Comparison
- direct IoU/F1/material/correct: 15.31% / 26.55% / 0.16% / 0.09%
- structured IoU/F1/material/correct: 19.90% / 33.20% / 46.79% / 30.80%

## Repair Effort
- direct normalized_edit_distance: 1.0830
- structured normalized_edit_distance: 1.0365
- direct edits(add/del/rep): 609 overlap ref, 2938/516/342
- structured edits(add/del/rep): 778 overlap ref, 2792/469/372

Diagnostic-only: this case summary is not part of the main/supplementary benchmark claims.
