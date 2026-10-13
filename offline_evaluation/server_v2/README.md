# Server-only evaluation v2

This namespace is `new_implementation`. It does not recreate missing ARR/fixed_builder code or claim numerical equivalence to published tables. Historical APIs (`Block`, `VoxelGrid`, fixed_builder modules) remain unavailable; legacy collection errors are preserved. Existing `tools` metrics coarsened some material identifiers and used overlap material-match rates; v2 preserves exact identifiers and uses labelled-set IoU, so their numbers are not interchangeable.

Storage arrays are Unicode NPY, axis Y,X,Z, with inclusive padded bbox bounds. World coordinates are `(xmin+xi,ymin+yi,zmin+zi)`. The storage origin is bbox minimum; generation metadata's anchor is a separate quantity. `air`, `cave_air`, and `void_air` mean absence. Loading rejects object dtype before `np.load(...,allow_pickle=False)`. Occupied dimensions differ from padded array dimensions.

The strict JSON action schema accepts a bare list or `{"operations":[...]}`. Operations are `set` with x/y/z/block, `fill` with x1/y1/z1/x2/y2/z2/block, and `carve` with the same endpoints and no block. Endpoints are inclusive; reversed endpoints are sorted as an explicit schema rule. Operations apply sequentially, the last write wins, and air/carve delete. Unknown fields, booleans as coordinates, excessive operations, cuboid visits, live voxels and coordinates fail. No eval, shell, truncation, rotation, scale, clamp, or GT-dependent ordinary-output repair occurs.

Metrics report Absolute and Aligned separately. Alignment exhaustively examines translations inside the recorded radius, maximizes occupancy overlap, and breaks ties by minimum L1 shift then lexicographic XYZ. Every aligned metric uses that one selected shift. It is exact inside the finite search cube, not a claim of globally optimal unrestricted alignment. Material-aware IoU uses `(position,exact material)` set intersection/union; a replacement contributes one unmatched item on each side. Add/Delete/Replace cost is normalized by `max(1,GT occupied cells)`. Both-empty overlap scores are one; failed or unexecuted cases have null metrics. Building-cluster bootstrap keeps paired rows within buildings, gives equal building weight, and excludes failures while reporting counts; descriptive CIs are not significance tests. Holm adjustment is implemented and tested for a future registered hypothesis family.

Conditions: Original uses saved description text only. Canonicalized is a **new lossless JSON key-order/whitespace transformation of that same saved description**, not historical ARR canonicalization. Partial Oracle explicitly exposes GT occupied dimensions, world bbox, exact material set and voxel count. Exact-spec Oracle losslessly encodes equal-material contiguous Z runs at fixed Y,X with a sorted palette. Missing coordinates decode to air; runs and palette decoding rules are explicit in the shared prompt. Executor control executes GT-derived run fills without a model. Oracle/control successes are implementation checks and never Builder successes. Oracle differences are diagnostic, not pure language information loss.

Cohort seed is 20261013. Rank quintiles within each v1/v4 dataset contribute one smoke and six evaluation buildings: 10 smoke and 60 evaluation, disjoint. This is a subset of existing data, not unseen data. Only 500 primary descriptions exist; v1 Gemini100 are absent. Evaluation has150 Original+150 Canonicalized inputs, plus60 Partial+60 Exact-spec once per building. Smoke has25+25 normal inputs, plus10+10 Oracle inputs. Executor control was run on all200 GT, not replicated by description model. Adding missing v1 Gemini would require a separately authorized **new cohort**, not manufactured historical data.

Create a dedicated venv and install `requirements.lock`; no system pip installation. The tested environment is recorded under reports/server_only. Commands below use the existing venv:

```bash
P=.venv-server-v2-20261013/bin/python
R=/root/minecraft-runs/server-only-20261013-001
$P -m offline_evaluation.server_v2.cli --help
$P -m offline_evaluation.server_v2.cli dry-run --run-dir "$R/builder_smoke_final"
$P -m offline_evaluation.server_v2.cli evaluate --run-dir "$R/builder_smoke_final" --alignment-radius 1
$P -m offline_evaluation.server_v2.cli summarize --run-dir "$R"
```

New CPU tasks use a new unique run directory. `prepare` freezes the subset before prediction scores; `verify-gt` independently checks all real GT and also creates Oracle inputs if that subset exists; `reevaluate` reads saved older i2t2b outputs. Completed GT/reevaluation summaries cannot be overwritten. Workers default2; set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1. Point TMPDIR inside the run directory for tests.

Generation is NOT_RUN in this Goal: zero research requests. OpenAI Responses and Anthropic Messages use separate HTTP adapters and the same inputs, prompt, validator, executor and metrics. No SDK retries or model fallback occur. `examples/second_builder_disabled.json` explicitly retains `claude-haiku-4-5-20251001`, api_allowed=false and zero budgets. A dry-run does not establish model availability or account access. API contracts were checked against official documentation on2026-10-13 (URLs in providers.py).

A future, separately authorized generation needs OPENAI_API_KEY or ANTHROPIC_API_KEY in its environment (never put keys into configs/artifacts), an explicit model and provider, api_allowed=true, `--allow-api`, positive max_calls/max_tokens/max_cost_usd, max_output_tokens, max_attempts_per_input, and a dated verified pricing/context record. Required pricing fields: verified=true, provider, model, verified_on, billing_mode=`standard_text_no_tools_no_cache_writes`, rate_scope=`upper_bound_full_context_window`, context_window_tokens, input_usd_per_million, output_usd_per_million, pricing_source_url and context_source_url. Official HTTPS sources, model identity, age (default7days), context ceiling and surcharge-covering upper rates are checked. Unknown pricing stops generation. Every request reserves the full context ceiling plus output ceiling, conservatively covering billing; unknown usage/timeouts preserve that reservation and halt. Costs round upward rather than undercount. Reservations and attempts persist across restart, including retry budgets. Budget limits can therefore stop earlier than simple average-token estimates.

Changing config, prompt, implementation source or inputs requires a **new Builder directory**; existing manifests are immutable. For a future operator, after creating an authorized enabled configuration at a scoped path:

```bash
$P -m offline_evaluation.server_v2.cli prepare-builder --run-dir "$R" --config /root/minecraft-runs/FUTURE_ID/enabled.json --split smoke --builder-name builder_future_smoke
$P -m offline_evaluation.server_v2.cli dry-run --run-dir "$R/builder_future_smoke"
# This command sends paid requests and was NOT RUN in Goal001C:
$P -m offline_evaluation.server_v2.cli generate --run-dir "$R/builder_future_smoke" --allow-api
$P -m offline_evaluation.server_v2.cli evaluate --run-dir "$R/builder_future_smoke" --alignment-radius 1
$P -m offline_evaluation.server_v2.cli summarize --run-dir "$R"
```

Prepared runs also bind the generation.py/providers.py/core.py source hashes; changed or missing implementation hashes block resume before transport. Older hashless preparations are preserved as superseded NOT_RUN artifacts; final Builder directories use the bound implementation. Report-only commits can change HEAD while these source hashes remain identical.

Raw/parsed/voxel/metric files are separate. Atomic fsynced writes, flock, UUID attempts, UTC timestamps, full input/config/prompt/commit/model identities and executing generation/provider/core source SHA256, response ID/usage and output hashes support resume. Tests inject mock HTTP only with fixture_only provenance. No mock output is claimed as a real API success.

Checkpoint creation requires a completed-task freeze, original source SHA checks, no active Builder lock, a path allowlist, and secret scans of actual files. Original source is archived once and later snapshots reference its immutable SHA. Archives reject links and unsafe paths; restore verifies every file hash in a new directory. The authorized GitHub scope is this dedicated branch and new draft releases only. Release upload/download is performed with the authenticated gh CLI; connected GitHub tools do not offer this complete asset lifecycle. Never publish, clobber assets, move tags or push main.
