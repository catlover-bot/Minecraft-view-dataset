# Goal005 official documentation evidence

Actual retrieval: `2026-10-13T21:24:22.644662Z` (server UTC); public documentation only. No research, model-discovery, or token-count API call occurred. The fixed model IDs and generation payloads remain unchanged. Account availability was not probed; the completed smoke's saved responses are historical account evidence only.

| Fixed model | Input / output USD per million tokens | Context | Model output maximum |
|---|---:|---:|---:|
| gpt-5-mini | 0.25 / 2.00 | 400000 | 128000 |
| claude-haiku-4-5-20251001 | 1.00 / 5.00 | 200000 | 64000 |

The OpenAI model page documents prices, limits, structured-output support and response alias/snapshot IDs. The model is marked deprecated; no automatic fallback is permitted. [OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-5-mini).

The Haiku 4.5 model page and pricing page document the unchanged standard first-party rates and limits. Its legacy designation does not authorize substitution. [Anthropic model documentation](https://platform.claude.com/docs/en/models/haiku-4-5/overview), [standard pricing](https://platform.claude.com/docs/en/about-claude/pricing).

The frozen adapters use Responses `text.format` with strict JSON schema and Messages `output_config.format`. Anthropic documents Haiku 4.5 support and a billable injected format prompt. Neither native format support nor schema success guarantees semantic material/geometry correctness. [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

OpenAI's output cap includes reasoning as well as visible output. A locally tokenized reference action body is therefore not a completion-length guarantee. [Reasoning token accounting](https://developers.openai.com/api/docs/guides/reasoning).

`official_pricing.json` records the actual retrieval and fixed-model evidence. The proposed new cumulative USD10 application limit is unapproved and is neither an invoice nor a promise to finish 160 requests. Full-context-plus-output reservations are deliberately separate from empirical smoke extrapolation. CPU history verification and disabled planning do not require a same-day pricing date; future paid starts/resumes do.
