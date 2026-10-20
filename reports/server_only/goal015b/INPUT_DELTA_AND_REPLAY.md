# Goal015B: old O versus current FULL inputs and saved-output replay

Both ID-selected old nonempty outputs remain nonempty through the frozen Goal015 evaluator. This confirms compatibility of these two saved outputs with the current extraction, validation, execution, and metric path. It does not explain why the later models returned deletion-only programs.

Selection was fixed to the nine currently sent scenes, the same description provider and Builder, then ascending scene ID. No IoU ranking was used. Both selected examples are building 003, using its saved OpenAI description. Each Builder had nine eligible old nonempty O examples.

| Evidence | Old OpenAI O | Old Anthropic O |
|---|---|---|
| Selected global slot | 428 | 429 |
| Operations / occupied voxels | 52 / 1856 | 57 / 1817 |
| Replay parse / exact saved grid / metric execution | PASS / PASS / PASS | PASS / PASS / PASS |

| Dimension | Old O | Current FULL (both repeats) |
|---|---|---|
| Summary and seven other semantic fields | All eight original typed values | All eight exactly equal to old O |
| Management metadata | `building`, `created_at`, `model`, `provider` present | All four omitted |
| Input serialization / bytes | Original formatted file, 3384 UTF-8 bytes | Sorted compact exact serializer, 2851 UTF-8 bytes, no final newline |
| Prompt | Same exact bytes | Same exact bytes |
| Native output schema | Same shared set/fill/carve schema; empty array permitted | Same exact typed schema |
| OpenAI requested / returned model | `gpt-5-mini` / `gpt-5-mini-2025-08-07` | Exact snapshot requested and returned |
| Anthropic requested / returned model | `claude-haiku-4-5-20251001` / same | Same exact snapshot requested and returned |
| OpenAI reasoning | Request omitted; response reports `medium` | Request explicitly `medium`; response reports `medium` |
| OpenAI sampling | Request temperature/top-p/seed omitted; effective defaults unknown | Same omitted request settings; effective defaults unknown |
| Anthropic sampling / thinking | Temperature/top-p/top-k/seed/thinking omitted; effective defaults unknown | Temperature 1.0 and thinking disabled explicit; top-p/top-k/seed omitted |
| Maximum output tokens | 32768, both Builders | 32768, both Builders |
| Service tier | OpenAI `default`; Anthropic `standard_only` | Same explicit requests |
| Generation time | 2026-10-14 UTC (exact timestamps in private receipt) | 2026-10-19 UTC (exact timestamps in private receipt) |
| Environment evidence | Bound generation commit/code; actual Python/package process environment not recorded in selected chain | Plan names existing Python environment and packages; actual historical process environment not captured by this comparison |

The prompt hash is `ba7a555860628c1ad55ba5b65200ac3cf0d528be86c42f46dbb5bbdd61e6ccac`. The old bound schema hash is `41a28cf6044541d67f794b5705b934749f94ef882732becb325d8cd1af2fdf83`; equality was checked on actual schema trees.

Goal007's manifest authenticates its canonical selected-slot product and original Goal006 plan/ledger. The original Goal005 full schedule agrees with the selected Goal006 input text. The original description bytes, raw provider envelope, parsed document and voxel artifact were each hash-checked before decoding. There is no separately saved old HTTP request artifact. The old request was reconstructed using only the frozen pure payload functions and matched the exact object hash in both ledger and selected slot:

- openai: `dde9d38d7548dbb46ab3ae7a16f3ca87948c770174c11a09b29e6ef72ceb651b`; old attempt `c8d15efee21c4e2780f0689cc64f5fda`; current FULL calls `goal015_call_011`, `goal015_call_014`.
- anthropic: `87fe938a0ae8d175e420371eaf8ad46db2002dc8e093a5c6c4a0ead73e0fa389`; old attempt `0b3bed83070649ebafb3a5e50308ff73`; current FULL calls `goal015_call_010`, `goal015_call_015`.

This provides saved request-construction provenance, not independent proof of the historical bytes on the wire. Omitted defaults remain unknown even when a response echoes a value. The removed metadata, formatting, explicit settings and five-day time difference are observable differences, not established causes.

Replay identifiers are `goal015b_replay_openai` and `goal015b_replay_anthropic`; they do not impersonate parent calls. Each independent raw-envelope extraction was passed unchanged into `Goal015.evaluate_one`. A separate check using the actual current `Goal015.adapter.parse_response` and `validate_output` accepted both old native raw envelopes, returned no failure, validated 52/57 operations, and produced the exact same text SHA as these evaluator replays. GT was explicitly the saved old prediction as a synthetic self-reference fixture, keeping the real evaluation path without loading or clipping actual GT. Both parsed documents and all saved coordinates/materials match exactly. Self-reference metrics (IoU and surface F1 equal 1) are fixture checks and are not new reconstruction results.

Replay computation: 1.090 CPU seconds, 1.090 wall seconds, peak RSS 125640 KiB, one thread. Additional research requests: 0. Original outputs, parent calls and the eight unsent calls were not modified.

The separate adapter parity check used 0.138 CPU seconds, 0.139 wall seconds and peak RSS 38400 KiB; it did not rerun metrics.

Private evidence: `phase4/INPUT_SNAPSHOTS.jsonl` (pre-read path/hash/size log), `phase4/COMPARISON_REPLAY.json` (selection frame, per-field hashes, request/response settings, timestamps, exact source descriptors and replay checks), and `phase4/CURRENT_ADAPTER_PARITY.json` (actual current native parser/schema parity) under the new Goal015B run.
