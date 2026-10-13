# GitHub checkpoint verification

Status: **GITHUB_VERIFIED / PARTIAL**. Draft Release [411682582](https://github.com/catlover-bot/Minecraft-view-dataset/releases/tag/untagged-c9b95fe3b02091d7ae2e) remains unpublished. Target commit `6bf1629cc4a0bca92d6890389148ae38c4b1a3ee` was pushed and remote-SHA verified. Code branch: [research/server-only-20261013](https://github.com/catlover-bot/Minecraft-view-dataset/tree/research/server-only-20261013).

All9 assets were downloaded from GitHub into a separate directory. All archive/metadata hashes match; safe extraction verified11029 original files and2137 snapshot files. A real-GT Oracle/executor smoke and both70/420-input disabled Builder manifests passed using downloaded/restored code. Research generation requests:0.

| Asset | Bytes | SHA256 |
|---|---:|---|
| checkpoint.json | 1208 | `4fa81ad8a173e1080615ce8afa488bba47fb2f2de62bc6a89c8c52966bb7b0e8` |
| checkpoint.tar.gz | 6779347 | `5853387e4e80f2788fa3b2e7bd631ee44499303bc1f6157ac3d5d8e8fb238e7e` |
| checkpoint.tar.gz.files.json | 483129 | `af8c359e08d5416719707876895220b106d3295c8cdf3dea6a227b1bc4222306` |
| local_restore_verification.json | 502 | `e95c6861120044f136e14d662ed6853489a6c7ef9e04a5a715dd8187b10b21bf` |
| RESTORE.md | 1242 | `b2da4f0c4eab47c1661fd9efd88d163bfecf8a7307108f898f23f2c80ec3a65d` |
| SHA256SUMS | 706 | `6a959dacf3fbd0e25de44caa4b30dcff698c7aaa5f6e100bb20d044697428694` |
| sources.tar.gz | 769781742 | `97b5d9f0862449c881cd5804485f1a27cc5164baf3f51df09c66c2658ac3d6da` |
| sources.tar.gz.files.json | 2783909 | `9c7e5878df1837c51ece9a03aa64c4a5b8cbe08352aaa71b9236d6c1e1a9cb65` |
| supplementary_secret_scan.json | 233 | `dbb623429ed1476aa5f12281d2b3e9301e72feb51b3a7c8f5fab79acee608f72` |

Measured archive171.65s / RSS86308KiB; upload57.69s /49920KiB; download82.23s /50048KiB. Additional restoration timing and final free bytes are in checkpoint_index.json. No estimates are presented as completed measurements.

The sourceasset is stored once. Later snapshots should use source_reference.json; never clobber existing assets/tags. Missing ARR modules/rawoutputs, Direct100/repeat30 originals, and v1Gemini100 remainBLOCKED. Old aggregation CSV matching is not raw-output replay. An alternative complete-v4 saved-description cohort plan is in alternative_cohort_plan.json, PLANNED_NOT_RUN.

No Git authentication or .git directory is bundled. CPU stages and existing Builder manifests work with restored code; fresh prepare-builder/checkpoint operations need a Git checkout. The initial session sandbox/approval settings were not enabled by this Goal and were not changed.

Commands below use the real verified CLI. The final Builder input count is420 including120 Oracle diagnostics; normal evaluation is300. Reusing a disabled run with --allow-api still cannot generate; an enabled, dated and budgeted configuration must be prepared in a new immutable directory first.

```bash
P=.venv-server-v2-20261013/bin/python
R=/root/minecraft-runs/server-only-20261013-001
$P -m offline_evaluation.server_v2.cli dry-run --run-dir "$R/builder_smoke_final"
$P -m offline_evaluation.server_v2.cli evaluate --run-dir "$R/builder_evaluation_final" --alignment-radius 1
$P -m offline_evaluation.server_v2.cli summarize --run-dir "$R"
# Choose a unique checkpoint ID; completed.json must still validate.
$P -m offline_evaluation.server_v2.cli checkpoint --run-dir "$R" --destination /root/minecraft-checkpoints/FUTURE_UNIQUE_ID --source-allowlist reports/server_only/source_archive_allowlist.json --source-reference reports/server_only/source_reference.json --checkpoint-id FUTURE_UNIQUE_ID
# Choose a NEW empty restore destination.
$P -m offline_evaluation.server_v2.cli restore --archive /root/minecraft-checkpoints/server-only-20261013-002/github_download/checkpoint.tar.gz --sha256 5853387e4e80f2788fa3b2e7bd631ee44499303bc1f6157ac3d5d8e8fb238e7e --destination /root/minecraft-checkpoints/FUTURE_NEW_RESTORE_DIR
```

Future paid generation, not executed in this Goal, requires explicit provider/model, keys in the environment, api_allowed=true, --allow-api, call/token/USD/output/retry ceilings and current verified context/pricing including surcharges. Prepare it with a new --builder-name, dry-run first, then use generate/evaluate/summarize. Exact commands and required pricing fields are in offline_evaluation/server_v2/README.md.
