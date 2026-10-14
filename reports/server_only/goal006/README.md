# Goal006: completed160 preserved; fixed suffix800 prepared without approval

Additional research API sends in this Goal: **0**. No paid model/token-count probes and no real TTY approval were performed. The active root session uses workspace-write and per-operation auto_review. Ordinary commands and workspace writes succeeded; this is not a claim of OS-level isolation. The initial HEAD was `25dad5b53554dc600fb5af005402b2853ab8d154`; the dedicated branch remains `research/server-only-20261013`. Main is expected to remain `7309327efd1c19dda793b19e8d8e6ce0b5868db9` and is checked against GitHub at each save.

## Completed measurements were saved first

All 160 attempts and claims correspond one-to-one with saved responses and evaluation rows:154 successful nonempty reconstructions,6 model-output failures,0 unknown usages,0 outstanding reservations,0 NOT_RUN. OpenAI has76/80 successful outputs (O29,J28,P10,E9); Anthropic78/80 (O30,J30,P10,E8). There were160 response/completed outcomes and157 natural-stop/JSON/schema successes. The143+17 interrupted run has no duplicate provider/input or repeat attempt.

The new160 consumed1,474,393 tokens and conservative cost USD3.36212875 with retry0. Including the earlier30 exactly once, cumulative usage is190 calls,retry4,1,828,481 tokens,USD4.048922. These are conservative accounting values, not a final invoice. Original logs contain the complete final JSON; the literal `BATCH_EXIT_CODE=0` marker is absent, so the user's reported exit code is distinguished from directly observed evidence.

The initial original validator pass checked154 raw→parsed→voxel chains. Its pinned proof is reused only while the complete641-file inventory and old implementation hashes match. Originals, approval, authority, claims, pricing revisions and summaries remain unchanged. Fixed8 representative metric dictionaries were recomputed once and matched; no154-metric rerun or200-GT/1200-output reevaluation was done.

Before continuation implementation, commit `97bde1ba5f84d1aceba65fe32e3d4b9eae5b2d11` and new Draft Release **412585766** preserved649 files, including full960manifest,160raw,154parsed/voxel pairs,160claims and environment. Archive asset **640015775**, SHA256 `571a8c1e5175c5238c2f75393c81557b8c735db4f5ceacb5217c5cf08125bb44`. All6assets were downloaded again and safely extracted, with hashes and complete response chains verified. [GitHub verification](results_github_verification.json) records IDs, URL and paths. Old source/30-response assets are immutable ID/SHA references and were not re-uploaded.

A secret-pattern collision in one OpenAI encrypted reasoning field was classified by exact raw SHA and field location, strict base64url/envelope validation, and scanning all other fields. Original bytes were retained. No matched token-like value is printed in reports. The classification is recorded in [archive_scan_classifications.json](archive_scan_classifications.json).

## Saved-result diagnosis

The3 validation failures are OpenAI J078 operation4 `wood (plank, oak)`, J062 operation2 `quartz (white)`, and O062 operation2 `sandstone (or smooth_sand)`, all outside the frozen material-ID regex. Truncations are Anthropic E032/E092 and OpenAI E092, each at32768 output tokens; OpenAI includes384 reasoning tokens. No normalization, cap extension or prompt/schema change was applied.

P's exact dimensions/bbox/material-set/voxel-count counts are OpenAI1/1/6/4 out of10 and Anthropic10/10/8/0. Neither provider satisfies all4attributes on any of its10P rows. Anthropic's count excess is+5101..39841, with saved fill/voxel evidence. E has9OpenAI-valid,8Anthropic-valid and8common-valid buildings; exact shape counts6/9 versus2/8, exact labelled voxels2/9 versus1/8. [Diagnosis](diagnosis/README.md) reports valid-only, all-attempt failure0 and common-valid means separately, with actual denominators. Results remain descriptive; J is JSON reserialization, P is GT attributes only, and E is the complete discrete GT specification.

## Fixed continuation and budget proposal

New code is isolated in `offline_evaluation/continuation_v1`; hash-bound batch_v1/resmoke_v2/server_v2 files remain unchanged. The plan takes `full_schedule[160:]` verbatim:50buildings,800provider/input pairs, global indices160..959, each provider O150/J150/P50/E50. There is no new selection or reorder. Five160-input segments are checkpoint/progress boundaries under one scope and cumulative budget.

Proposed limits, **not yet approved**: new800calls/retry0/concurrency1/output32768; cumulative990calls/retry4/USD30. New token reservation ceiling266,214,400; cumulative268,042,881. Full-context input plus output is reserved durably before each request. Unknown usage retains its reservation and stops without retry. Known model-output failures continue; operational/identity failures stop. Approval requires an exact plan-bound phrase from a real TTY, separate from the old190-call approval.

Composition-weighted extrapolation from160actual attempts is **USD16.81064375 additional, USD20.85956575 cumulative,10.0817965 active hours**. Observed160active time is2.0163593h; calendar span11.2419822h includes a9.2251382h UTC-pricing pause excluded from processing time. Summing all800 maximum individual reservation envelopes givesUSD211.7504; that is neither prepayment nor an expected bill. USD30 may stop before all800 complete. [Cost cells](planning/costs/costs.json) retain provider×condition×description-source usage and duration.

Official evidence actually retrieved on2026-10-14 at08:48:58Z is reused without changing its date. [Evidence reuse](official_evidence_reuse.json) records8official source documents. Actual availability/account probes were not sent. Before each future request, same-day freshness is checked; actual refreshed evidence is appended to the same plan history without resetting approval or ledger.

The absolute launcher, preview, status, evaluation, pricing-refresh, checkpoint-upload retry and fixed960 aggregate commands are in [OPERATIONS.md](OPERATIONS.md). Final960 analysis uses existing building-level paired bootstrap/sign-flip/Holm families8/4/8 only when all planned outcomes are complete. Final aggregation was tested with fixtures only; no results were fabricated for the unexecuted800.

## Validation and saved plan

[Tests](test_summary.json): **220unique related cases PASS**, final0fail/error/skip (180existing,11diagnosis,29continuation). Initial failures and their targeted corrections are retained:40existing cases initially used a disallowed/tmp output location;1new case had a fixture-only str/Path mismatch. Only affected cases were rerun, plus3aggregate cases after an actual denominator-table change. The old6collection errors remain referenced in Goal003 evidence.

The next report records the actual disabled canonical plan, identity/SHA, preview and independent Draft archive after the bound code commit has been pushed. No enabled directory, authority or reservations are created by preparation. Restored copies cannot activate the paid scope.
