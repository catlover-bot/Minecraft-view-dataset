# Saved prediction validation failures

The new reader evaluated 1200 saved I2T2B predictions: 1168 were compatible and 32 were rejected. All 32 rejected arrays have safe string NPY headers and shapes matching their inclusive `Y,X,Z` bboxes. The rejection concerns stored material labels, not axes or bbox geometry.

Provenance: `new_implementation_reader_validation_of_historical_artifacts`. Original repository: `catlover-bot/Minecraft-view-dataset`; original ref: `7309327efd1c19dda793b19e8d8e6ce0b5868db9`. Historical ARR replay remains unverified. Research API requests: **0**.

There are 13 affected buildings and 11 distinct incompatible labels. The saved renderer accepts descriptive tokens such as `quartz_(or_white_block)` and `polished_andesite|smooth_stone`; these do not satisfy the new executable identifier schema. Two artifacts also confirm that `<U32` storage truncated `sandstone_(or_light_yellow_block)` to its first 32 characters, as verified against their saved actions.

The reader is unchanged. Original files were not modified, clamped, or remapped. These rows retain `metrics=null`; they must not be assigned favorable zero errors or included as valid reconstruction results. This reader rejection does not establish that historical Minecraft execution failed: no Minecraft runtime was used.

Only the 32 failed records were revalidated. Every voxel, bbox and action source was checked against the original-file manifest SHA-256 and verified Git blob correspondence. [Detailed JSON](reevaluation_failures.json) records every path, original ref/blob/SHA-256, bounded header, bbox, invalid-label count, example coordinate and confirmed truncation.

| Scene | Rejected saved predictions |
| --- | ---: |
| `buildings_100_v1_building_043` | 3 |
| `buildings_100_v1_building_054` | 3 |
| `buildings_100_v1_building_088` | 3 |
| `buildings_100_v1_building_092` | 1 |
| `buildings_100_v1_building_097` | 3 |
| `buildings_100_v4_building_009` | 2 |
| `buildings_100_v4_building_034` | 2 |
| `buildings_100_v4_building_053` | 2 |
| `buildings_100_v4_building_060` | 2 |
| `buildings_100_v4_building_062` | 3 |
| `buildings_100_v4_building_068` | 3 |
| `buildings_100_v4_building_078` | 2 |
| `buildings_100_v4_building_084` | 3 |

| Incompatible stored label | Voxel occurrences across rejected artifacts |
| --- | ---: |
| `dark_prismarine?_(nether_brick?)` | 978 |
| `nether_brick_(dark)` | 419 |
| `polished_andesite|smooth_stone` | 8 |
| `quartz_(or_white_block)` | 8995 |
| `quartz_(or_white_concrete)` | 24 |
| `quartz_(or_white_stone)` | 1105 |
| `quartz_(white)` | 6764 |
| `quartz_(white_block)` | 573 |
| `sandstone_(or_light_yellow_block` | 6978 |
| `sandstone_(or_smooth_sand)` | 6794 |
| `wood_(dark_oak_logs)` | 1458 |

The legacy renderer creates `<U32` arrays and writes normalized strings in `tools/render_rebuild_from_plan.py:146`; its normalization is imported from `tools/evaluate_rebuild_metrics.py`. This audit records those source hashes without importing or executing the legacy evaluator.
