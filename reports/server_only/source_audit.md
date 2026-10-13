# Original-source audit

Original repository: https://github.com/catlover-bot/Minecraft-view-dataset; original ref `7309327efd1c19dda793b19e8d8e6ce0b5868db9`.
All 25 reachable commit trees were checked. The clone is not shallow but is a promisor partial clone (`blob:none`). Tree inspection and existing local-file hashing avoid indiscriminate retrieval. GitHub metadata showed only `main`, with no tags, Releases or Actions artifacts at audit time. No LFS or submodule declaration, gitlink or current pointer was found. No missing historical source was recovered.

| Artifact | Expected | Found | Status |
|---|---:|---:|---|
| Primary GT scenes | 200 | 200 | FOUND |
| GT image files | 2400 metadata references | 2400 | FOUND |
| Descriptions | 600 | 500 | PARTIAL |
| ARR fixed Builder outputs | 1200 | 0 | MISSING |
| ARR canonicalized inputs | corresponding saved inputs | 0 | MISSING |
| ARR recorded attempts / successful actions / predicted voxels | separate raw artifacts | 0 / 0 / 0 | MISSING |
| ARR Direct raw outputs | 100 | 0 | MISSING |
| ARR stochastic scene IDs and runs | 30 buildings | 0 | MISSING |
| Older i2t2b action files / voxel files | separate older experiment | 1200 / 1200 | FOUND |

Descriptions are OpenAI 200, Claude 200, Gemini 100 (v4 only). No v1 Gemini description exists in a reachable commit tree. Older i2t2b outputs comprise two description providers and three schema/material/self-refinement variants; they cannot substitute for ARR's three models and two representations. Their generation-attempt count and confirmed successful API-call count are unknown; 1200 stored action files contain parseable operation lists.

Two supplementary LLM-authored Minecraft cohorts each contain 10 GT scenes, 10 descriptions, 20 action files and 30 voxel files. They are recorded separately and excluded from the primary counts and source archive allowlist. The manifest still inventories their already present original files.

`offline_evaluation/fixed_builder/`, `executor/`, `metrics.py`/`metrics/`, `experiments/fixed_builder_v1/` and the three imported v2 analysis/coverage modules are absent throughout reachable history. The existing second Builder script only prepares a dry-run manifest. The only old requirements file lists NumPy and Pillow, omitting analysis/test dependencies.

ARR saved tables contain 600 main paired rows over 200 buildings and three models, 700 Direct comparison rows over 100 buildings and seven conditions (including 100 Direct rows), and 36 stochastic aggregate rows. Older rebuild metric JSONs contain 2200 saved evaluation rows from multiple variants. These are saved-table evidence, not reproduction from generated ARR outputs. ARR historical replay is BLOCKED. CPU reevaluation of the 200 real GT and 1200 older artifacts with `new_implementation` metrics remains possible.

Every locally present file from the original tree is recorded in `originals_manifest.jsonl` with repository, ref, relative path, Git blob hash, SHA-256 and size. `17339` files / `3289333962` bytes were hashed; worktree mismatches: 1. `available_predictions.json` provides scene/condition/action/voxel sources. `source_archive_allowlist.json` selects the immutable primary data and older output sources for a separately scanned archive.

All six legacy test modules (53 test functions) were reviewed statically and are API-free; they require NumPy, pandas, statsmodels and pytest. Collection currently depends on absent original modules. Actual pass/fail/skip/error counts are recorded by the root agent's test run. Some old tests only assert constants or file presence and do not establish robust resume or API behavior.

No research API was called, no pickle was loaded, and no raw NumPy payload was loaded by this audit. NPY headers are Unicode arrays. The original GT bbox specifies `Y,X,Z`, inclusive bbox endpoints and world coordinates `(xmin+x_index, ymin+y_index, zmin+z_index)`; the historical evaluator's permissive `allow_pickle=True` need not be used.
