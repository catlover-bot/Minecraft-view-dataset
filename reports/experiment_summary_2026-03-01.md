# I2T2B Experiment Summary (2026-03-01)

## Scope
- Task: existing-building image -> description -> rebuild plan -> voxel rebuild -> GT comparison.
- Datasets: `buildings_100_v1`, `buildings_100_v4`.
- LLMs: OpenAI `gpt-5-mini`, Anthropic `claude-haiku-4-5-20251001`.
- Rebuild metrics are the latest rerun after schema-repair + rebuild-world re-render.

## 1) Rebuild Quality (Latest)

| Dataset / Model | IoU | F1 | material_match | coarse_material_match | all_levels_pass |
|---|---:|---:|---:|---:|---:|
| v1 / OpenAI | 0.2467 | 0.3793 | 0.1817 | 0.3429 | 0.04 |
| v1 / Claude | 0.2280 | 0.3661 | 0.1512 | 0.3639 | 0.00 |
| v4 / OpenAI | 0.1624 | 0.2726 | 0.1783 | 0.3360 | 0.00 |
| v4 / Claude | 0.1695 | 0.2854 | 0.1758 | 0.3292 | 0.00 |

### Pass counts (out of 100)
- v1/openai: Level0=89, Level1=17, Level2=23, Level3=15, Level4=85, All-levels=4
- v1/claude: Level0=100, Level1=5, Level2=15, Level3=4, Level4=56, All-levels=0
- v4/openai: Level0=96, Level1=1, Level2=16, Level3=13, Level4=0, All-levels=0
- v4/claude: Level0=100, Level1=1, Level2=11, Level3=10, Level4=0, All-levels=0

## 2) Description Quality

| Dataset / Model | auto_score_mean | strict_material_f1 | coarse_material_f1 | dimension_score | completeness |
|---|---:|---:|---:|---:|---:|
| v1 / OpenAI | 0.8102 | 0.7269 | 0.9138 | 0.6547 | 1.0000 |
| v1 / Claude | 0.7202 | 0.5714 | 0.7295 | 0.6654 | 1.0000 |
| v4 / OpenAI | 0.7520 | 0.6146 | 0.8658 | 0.6047 | 1.0000 |
| v4 / Claude | 0.6893 | 0.5707 | 0.8089 | 0.4634 | 1.0000 |

## 3) Baseline vs Schema-Repair Rebuild (Delta)

| Dataset / Model | Delta IoU | Delta F1 | Delta material_match |
|---|---:|---:|---:|
| v1 / OpenAI | -0.0154 | -0.0222 | -0.0558 |
| v1 / Claude | -0.0659 | -0.0831 | -0.1353 |
| v4 / OpenAI | +0.0109 | +0.0146 | -0.1321 |
| v4 / Claude | +0.0167 | +0.0243 | -0.1465 |

Interpretation:
- v4 was slightly better in shape (IoU/F1), but strict/coarse material match dropped across all four settings.
- So overall reconstruction fidelity did not improve yet.

## 4) Pipeline Stability Improvement

Fallback (empty operations -> heuristic) after repair:
- v1/openai: `0/100` (repaired plans: 64, draft-salvage: 7)
- v1/claude: `0/100` (repaired plans: 99, draft-salvage: 0)
- v4/openai: `0/100` (repaired plans: 95, draft-salvage: 35)
- v4/claude: `0/100` (repaired plans: 100, draft-salvage: 1)

Interpretation:
- Major robustness gain: fallback-dominant failure mode is removed.
- Remaining bottleneck is plan semantics/material assignment, not parser failure.

## 5) Output Files (Latest Re-evaluation)

- `datasets/buildings_100_v1/metrics_levels_pe_v2_openai_gpt_5_mini_schema_repair_20260301.json`
- `datasets/buildings_100_v1/metrics_levels_pe_v2_anthropic_claude_haiku_4_5_20251001_schema_repair_20260301.json`
- `datasets/buildings_100_v4/metrics_levels_pe_v2_openai_gpt_5_mini_schema_repair_20260301.json`
- `datasets/buildings_100_v4/metrics_levels_pe_v2_anthropic_claude_haiku_4_5_20251001_schema_repair_20260301.json`

## 6) Recommended Next Step

- Focus prompt/control on material grounding in `rebuild_plan`:
  - add explicit material budget constraints,
  - add per-face material assignment checks,
  - add post-plan material sanity critic before rendering.
