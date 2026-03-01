# 再建築評価（IoU / F1 など）の読み方ガイド

更新日: 2026-03-01  
対象: `tools/evaluate_rebuild_metrics.py` による評価

## 1. 何を評価しているか

この評価は、GT（正解建築）と再建築結果（LLM plan からレンダリングした voxel）を比較して、次を測ります。

1. 形がどれだけ一致しているか（IoU / F1）
2. 材質がどれだけ一致しているか（strict / coarse）
3. 構造単位（平面連結成分）がどれだけ一致しているか（component_f1）

評価の実装:
- `occupancy_metrics`: [tools/evaluate_rebuild_metrics.py#L385](/Users/hirotaka-m/Minecraft-view-dataset/tools/evaluate_rebuild_metrics.py#L385)
- `material_metrics`: [tools/evaluate_rebuild_metrics.py#L402](/Users/hirotaka-m/Minecraft-view-dataset/tools/evaluate_rebuild_metrics.py#L402)
- `component_match_metrics`: [tools/evaluate_rebuild_metrics.py#L450](/Users/hirotaka-m/Minecraft-view-dataset/tools/evaluate_rebuild_metrics.py#L450)
- レベル判定: [tools/evaluate_rebuild_metrics.py#L521](/Users/hirotaka-m/Minecraft-view-dataset/tools/evaluate_rebuild_metrics.py#L521)

## 2. 評価前の位置合わせ（shift探索）

まず Pred を GT に平行移動して、最も重なる `dx,dy,dz` を探索します。

- 探索範囲（デフォルト）
  - `dx,dz`: ±48
  - `dy`: ±8
- 2D footprint の重なり上位候補を取り、3D intersection が最大になる shift を選びます。

実装:
- `search_best_shift`: [tools/evaluate_rebuild_metrics.py#L326](/Users/hirotaka-m/Minecraft-view-dataset/tools/evaluate_rebuild_metrics.py#L326)

意味:
- 「場所が少しズレただけ」で不当に低スコアにならないようにする処理です。

## 3. 各指標の意味（直感）

### 3.1 形状一致（非air occupancy）

- `precision = intersection / |Pred|`
  - Predで置いたブロックのうち、正しい場所の割合
- `recall = intersection / |GT|`
  - GTに必要なブロックのうち、再現できた割合
- `F1 = 2PR/(P+R)`
  - precisionとrecallのバランス
- `IoU = intersection / union`
  - 完全一致により厳しい形状指標

直感:
- `IoU` は厳しめ、`F1` はやや緩め。
- 形状が似ていれば両方上がるが、過剰生成や欠損で下がる。

### 3.2 材質一致

- `material_match`
  - GTとPredが同じ座標で重なった点だけ見て、材質IDが完全一致した割合
- `coarse_material_match`
  - 同じく重なり座標のみで、粗カテゴリ（STONE/WOOD/BRICK等）が一致した割合

重要:
- 材質は「重なった座標上」で評価されます。
- つまり形がズレると材質評価も連鎖的に下がります。

### 3.3 構造単位一致（component_f1）

- GT/PREDの平面footprint（x,z）から連結成分を抽出
- 成分同士のIoUが閾値（デフォルト0.30）以上なら対応付け
- その対応数で precision/recall/F1 を計算

直感:
- 大きな塊（棟や翼）が何個あって、どれだけ対応しているかを見る指標。

## 4. レベル別判定（Pass/Fail）

デフォルト閾値:
- `Level0`: intersection >= 32
- `Level1`: IoU >= 0.35 かつ F1 >= 0.50
- `Level2`: coarse_material_match >= 0.55
- `Level3`: material_match >= 0.40
- `Level4`: component_f1 >= 0.45
- `all_levels_pass`: Level0〜4をすべて満たす

実装:
- `DEFAULT_THRESHOLDS`: [tools/evaluate_rebuild_metrics.py#L18](/Users/hirotaka-m/Minecraft-view-dataset/tools/evaluate_rebuild_metrics.py#L18)

## 5. 最新結果（schema repair反映後）

参照ファイル:
- `datasets/buildings_100_v1/metrics_levels_pe_v2_openai_gpt_5_mini_schema_repair_20260301.json`
- `datasets/buildings_100_v1/metrics_levels_pe_v2_anthropic_claude_haiku_4_5_20251001_schema_repair_20260301.json`
- `datasets/buildings_100_v4/metrics_levels_pe_v2_openai_gpt_5_mini_schema_repair_20260301.json`
- `datasets/buildings_100_v4/metrics_levels_pe_v2_anthropic_claude_haiku_4_5_20251001_schema_repair_20260301.json`

平均メトリクス:

| 条件 | IoU | F1 | material_match | coarse_material_match | all_levels_pass |
|---|---:|---:|---:|---:|---:|
| v1 / OpenAI | 0.2467 | 0.3793 | 0.1817 | 0.3429 | 0.04 |
| v1 / Claude | 0.2280 | 0.3661 | 0.1512 | 0.3639 | 0.00 |
| v4 / OpenAI | 0.1624 | 0.2726 | 0.1783 | 0.3360 | 0.00 |
| v4 / Claude | 0.1695 | 0.2854 | 0.1758 | 0.3292 | 0.00 |

レベル通過件数（100件中）:

| 条件 | L0 | L1 | L2 | L3 | L4 | All |
|---|---:|---:|---:|---:|---:|---:|
| v1 / OpenAI | 89 | 17 | 23 | 15 | 85 | 4 |
| v1 / Claude | 100 | 5 | 15 | 4 | 56 | 0 |
| v4 / OpenAI | 96 | 1 | 16 | 13 | 0 | 0 |
| v4 / Claude | 100 | 1 | 11 | 10 | 0 | 0 |

規模傾向（平均 `pred_non_air_after_shift / gt_non_air`）:
- v1/OpenAI: 1.297（作りすぎ気味）
- v1/Claude: 0.778（不足気味）
- v4/OpenAI: 0.654（不足気味）
- v4/Claude: 0.735（不足気味）

## 6. どう解釈すればよいか

1. まず `L0` を見る
- ここが低いと、そもそも位置合わせ後も重なりが少ない。

2. 次に `IoU/F1`（L1）を見る
- 形の再現力の中心指標。
- 今回はここがボトルネック（特に v4）。

3. その後に `material_match`（L3）を見る
- 形がある程度合って初めて伸びやすい。
- いまは材質厳密一致が低く、再現の質を押し下げています。

4. `component_f1`（L4）は補助的に見る
- 棟数や塊の対応を見るので、大まかな構造把握には有効。
- v4では L4 が 0 で、構造単位の対応が崩れています。

## 7. 具体例（building_089, v1/OpenAI）

例: `datasets/buildings_100_v1/metrics_levels_pe_v2_openai_gpt_5_mini_schema_repair_20260301.json` の `building_089`

- shift: `dx=0, dy=3, dz=-2`
- IoU: `0.3608`, F1: `0.5302`（L1は通過）
- material_match: `0.3322`（L3は未達）
- coarse_material_match: `0.4966`（L2も未達）
- component_f1: `1.0`（L4は通過）
- 結果: `all_levels_pass = False`

読み方:
- 形は比較的良いが、材質指定が閾値に届かず総合不合格。

## 8. 実行コマンド（再評価）

```bash
python3 tools/evaluate_rebuild_metrics.py \
  --gt_root datasets/buildings_100_v1 \
  --pred_root datasets/buildings_100_v1 \
  --pred_source rebuild_world \
  --pred_subdir rebuild_world_pe_v2_openai_gpt_5_mini \
  --thresholds_json tools/thresholds_levels.example.json \
  --out datasets/buildings_100_v1/metrics_levels_pe_v2_openai_gpt_5_mini_schema_repair_20260301.json
```

同様に `pred_subdir` と `out` をモデルごとに切り替えて評価します。
