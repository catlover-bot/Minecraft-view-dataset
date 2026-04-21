# Human Reconstruction Protocol (LLM-authored 10-case diagnostic)

This package is protocol/toolkit only. No human results are claimed here.

## Task
Reconstruct each source building from provided images under one of the conditions:
1. image_only
2. image_plus_description
3. image_plus_description_plus_structured_ir

## Constraints
- Use provided allowed block list per case.
- Build inside provided build_area (bbox).
- Recommended time limit: 25 minutes per case.

## Submission format
For each case, submit:
- bbox.json
- voxels.npy
Path template:
- submissions/<participant_id>/<condition>/<case_id>/

## Scoring
Use `tools/evaluate_human_rebuild_submissions.py` to score with the same rebuild metrics framework.
