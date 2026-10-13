# Goal005 first evaluation batch operations

The completed16 resmoke is already archived and GitHub-redownload verified. This document concerns the separate **disabled** first evaluation batch. No Goal005 paid request or real confirmation has occurred.

Canonical pending plan: `/root/minecraft-runs/goal005-batch1-20261013T213420Z/plan.json`.
File SHA256: `231b6daffb671511cd3adc22847d9d72c26a37db14b5c6793520a6f987001b13`.
Identity: `f54ebfaa041a18cbbfb60d2d27f9e9c4a50d436545c7ed1175c60751be58816f`.
Implementation commit: `dd4925257cbbdfb88ac343c6c8d29483a69fa364`.

## Start and resume

In the existing root server terminal that contains both provider keys, use this single command (cwd-independent):

```bash
/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python -B /root/workspace/Minecraft-view-dataset/offline_evaluation/batch_v1/launch.py --plan /root/minecraft-runs/goal005-batch1-20261013T213420Z/plan.json
```

It displays all targets, fixed models and conditions, prior/current consumption, caps and destinations. Only a newline-terminated exact `/dev/tty` match enables a new run:

```text
APPROVE GOAL005 f54ebfaa041a 160 SENDS 0 RETRIES USD10 TOTAL
```

Codex has not entered this string into the actual launcher. Empty input, mismatch, EOF, Ctrl-C and terminal I/O failure do not activate. For a read-only preview, append `--preview` to the same command; it requires no keys or approval and sends nothing. The absolute entry was actually tested from `/tmp` with socket connections blocked.

Scope is fixed10 evaluation buildings and160 requests, concurrency1, retry0, output32768, with the completed prior30 calls/retry4/354088 tokens/USD0.68679325 referenced exactly once. Proposed cumulative maximum is190 calls/retry4/53596968 tokens/USD10. New token ceiling53242880 derives from80 requests per provider and full context400000/200000 plus output32768. The remaining800 requests have no execution permission. Budget shortage stops before a send; the application never raises a cap automatically. Full-context reservation is checked one request at a time, with known actual usage settled afterward; USD10 does not promise all160 will finish.

Use the identical command to resume the same original ledger. Settled attempted inputs are not retried, and unknown-usage or interrupted reservations stop without resend. Never delete/rewrite authority, approval, ledger or reservation claims; never prepare another plan to reset the lineage budget. Copied archival plans cannot activate. A partially interrupted activation also stops for investigation rather than recreating controls.

Exit0 means all160 observations finished with known usage and no operational halt, and does not imply160 valid reconstructions. Exit4 denotes a partial run. Exit2 is explicit non-approval, exit3 missing keys, exit5 a validation/operational stop. Model parse/validation/execution failures with known usage remain recorded outcomes; major provider/identity failures and unknown usage halt. Raw/usage/reservation/ledger are persisted after each attempt. This is application coordination and conservative accounting, not a billing guarantee or OS isolation claim.

## Later UTC day and CPU evaluation

Historical parent checks, disabled planning, preview and CPU evaluation remain possible when prior documentation is older. A future activation and each new paid send require today's real UTC official verification. To refresh, actually fetch the fixed models' official model/pricing/structured-output documents listed in `official_support.md`, retain the retrieved evidence and create a **new** complete JSON in the format of `official_pricing.json` with its actual UTC retrieval timestamp. Never edit the original frozen pricing file, fake a date, or disable freshness.

Use the existing CLI `python -B -m offline_evaluation.batch_v1.cli refresh-pricing --plan ORIGINAL_CANONICAL_PLAN --pricing NEW_OFFICIAL_RECORD` from the repository. It appends an immutable, hash-linked record in the canonical preparation's `pricing-revisions/`. Only the verification date and descriptive lifecycle status can differ in model records; prices, context/output limits, returned-ID allowlists, reservation/billing fields remain identical. Changes to those terms stop for an explicit scope decision. This narrow refresh does not edit the original plan, existing approval, ledger, authority or budget. Then resume with the same original start command.

For partial or completed batch CPU evaluation, the CLI `evaluate --plan ORIGINAL_CANONICAL_PLAN` can regenerate evaluation/summary from saved artifacts without same-day pricing. A disabled plan is rejected before creating an enabled directory. These derived batch reports may be refreshed; the completed historical resmoke originals are immutable and remain untouched. No alias correction, GT repair, target substitution or model fallback is applied.

## Interpretation and continuation

Batch1 is the first10 of the original fixed60 evaluation set, not another tuning smoke. Keep the frozen generation conditions when later authorizing the remaining50/800. If conditions change afterward, record a separate version and do not pool the results as one unchanged experiment.

Nine of60 reference action bodies exceed32768 local OpenAI tokens, including032 and092 in batch1. The measured reference is not a theoretical minimum or an actual-output guarantee. The current plan retains all60 and cap32768, recording actual truncation as a resource-constrained outcome if it occurs. A higher cap or alternate representation would be a separate explicitly decided condition. No large building was dropped or truncated during preparation.

The two-building, condition-weighted extrapolation is approximately USD3.1722475/2.18 hours for160 and USD19.033485/13.09 hours for960, with substantial building/description/output-length uncertainty. All-context reservation totals USD42.350080/USD254.100480 are separate conservative bounds, not completion estimates. Exact Anthropic token counts remain unknown; no token-count API was called.
