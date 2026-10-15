# Goal006A official documentation refresh, 2026-10-15 UTC

Actual clock measurements bracket public documentation retrieval at 11:13:29–11:14:51 UTC. Eight official HTML pages were opened and relevant sections inspected. Three OpenAI Markdown URL fetches returned unsupported-content-type errors; the corresponding HTML fetches succeeded. No model-list, token-count, or generation API was called.

The fixed models, standard first-party token prices, context and output limits, accepted model IDs, native schema routes, and conservative reservation terms match the frozen canonical plan. This is documentation evidence, not a new account entitlement or schema acceptance probe. The provider records retain historical account wording because the frozen refresh validator allows only verification date and lifecycle text to change; current probe status is explicit at document level.

- OpenAI `gpt-5-mini`: input USD0.25 and output USD2 per million tokens, 400,000 context and 128,000 output limit; structured outputs supported. [Model](https://developers.openai.com/api/docs/models/gpt-5-mini)
- The snapshot remains deprecated with shutdown scheduled for 2026-12-11. No model substitution. [Deprecations](https://developers.openai.com/api/docs/deprecations)
- Responses `text.format` retains JSON Schema and strict mode. Reasoning tokens count toward output usage and `max_output_tokens`. [Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [Reasoning](https://developers.openai.com/api/docs/guides/reasoning)
- Anthropic `claude-haiku-4-5-20251001`: USD1 input and USD5 output per million tokens, 200,000 context and 64,000 output limit. [Model](https://platform.claude.com/docs/en/models/haiku-4-5/overview), [Pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- Haiku 4.5 remains Active in the lifecycle table, with no deprecation date. The stated retirement lower bound is 2026-10-15; reaching this date does not establish a scheduled shutdown. [Lifecycle](https://platform.claude.com/docs/en/about-claude/model-deprecations)
- Messages `output_config.format` supports Haiku 4.5 without the old beta header. The additional format system prompt remains billable; these fetched pages do not publish a numeric overhead ceiling. [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)

`official_pricing.json` is ready for the existing continuation `refresh-pricing` operation. This preparation does not modify the canonical plan, approval, authority, ledger, experiment responses or pricing revision directory. The parent agent owns the separate append operation and its evidence.
