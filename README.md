# Minecraft + Malmo: One Building Multi-View Capture (Mac)

このプロジェクトは、Malmo 上に **単一の建築物** を生成し、見やすい距離・角度の多視点画像とGTを保存する最小構成です。

## 前提環境 (Mac)
- Java 8 (`JAVA_HOME` が有効)
- Malmo がローカルに配置済み
- Python 3.9+ 推奨

## セットアップ
1. Python 依存をインストール:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Malmo 環境変数を設定（例）:

```bash
export MALMO_DIR="$HOME/MalmoPlatform"
export JAVA_HOME="$(
  /usr/libexec/java_home -v 1.8
)"
source scripts/malmo_env_mac.sh
```

`MALMO_XSD_PATH` と `PYTHONPATH` は `scripts/malmo_env_mac.sh` が自動補完。

## ワンコマンド実行

```bash
./scripts/run_one_building_capture.sh \
  --out datasets/one_building_v1 \
  --port 10000 \
  --views 12 \
  --image_size 960 540 \
  --fov 70 \
  --seed 1234
```

## 100建築物を一括生成して撮影

```bash
./scripts/run_many_buildings_capture.sh \
  --out datasets/buildings_100_v1 \
  --count 100 \
  --start_style_id 0 \
  --port 10000 \
  --views 12 \
  --image_size 960 540 \
  --fov 70 \
  --seed 1234 \
  --spacing 180
```

出力は `datasets/buildings_100_v1/building_000` 〜 `building_099` に保存されます（各ディレクトリに `images/`, `gt/`, `meta.json`）。

## 出力

```text
datasets/one_building_v1/
  images/
    rgb_view00_yaw..._pitch....png
    ...
  gt/
    bbox.json
    voxels.npy
  logs/
    capture.log
  meta.json
```

- `meta.json` には `seed`, `created_at`, `bbox`, `palette/style`, `views[]`（`x,y,z,yaw,pitch,fov,width,height,path`）を保存。
- `gt/voxels.npy` は bbox 内のブロック名配列（shape: `[Y, X, Z]`）。

## 実装ポイント
- 遠すぎ問題: `bbox` の `width/depth` から撮影半径を自動計算し `clamp`。
- 生成前撮影問題: bbox 内 `non-air` ブロック数の安定判定（連続 K 回一致）で撮影開始。
- `client not listening :10000`: 起動スクリプト + ポート待機 + タイムアウト時診断。

## トラブルシュート

### 1) `client not listening on port 10000`
- まずログを確認:
  - `logs/malmo_client.log`
  - `datasets/.../logs/capture.log`
- 典型原因:
  - Java 8 ではない (`java -version` を確認)
  - `MALMO_DIR` が誤っている
  - `PYTHONPATH` に Malmo Python モジュールがない

確認コマンド:

```bash
echo "$MALMO_DIR"
echo "$JAVA_HOME"
python3 -c "import MalmoPython; print('MalmoPython OK')"
```

### 2) 安定判定で失敗する
- `datasets/.../logs/capture.log` の `stability sample` 行を確認し、`non-air` が変動し続けていないか確認。
- 建築 bbox が広すぎる場合は `building/generator.py` のサイズを下げる。

### 3) 画像が小さく見える
- `--fov` を小さめ（例: `60`）にする
- `tools/capture_one_building.py` の `compute_capture_radius` の係数（`1.2`, `margin`）を調整

## 次実験: image-to-text-to-build 評価
以下の評価CLIを追加しました:

- `tools/evaluate_rebuild_metrics.py`

### 目的
- GT建築 (`building_xxx/gt`) と再建築結果 (`building_xxx/rebuild_world`) を比較
- `TYPE_NORM` を強化して表記ゆれを吸収
  - 例: `stone_bricks`, `stone bricks`, `minecraft:stone_bricks` -> `stonebrick`
  - 例: `oak_fence`, `wooden_fence` -> `fence`
  - 例: `oak_planks` -> `wood`
  - 例: `stone_slab`, `stone_slab2` -> `slab_stone`
- `material_match` を intersection 上の一致率として算出

### レベル別評価 (metrics_levels.json)
- Level 0: 位置合わせ（shift探索）
- Level 1: 形状一致（IoU / F1）
- Level 2: 粗材質一致（STONE / WOOD / BRICK など）
- Level 3: 厳密材質一致（正規化後ID）
- Level 4: 構造単位一致（2D connected components の matching）

### 実行例
```bash
python3 tools/evaluate_rebuild_metrics.py \
  --gt_root datasets/buildings_100_v1 \
  --pred_root datasets/buildings_100_v1 \
  --pred_source rebuild_world \
  --thresholds_json tools/thresholds_levels.example.json \
  --allow_gt_fallback \
  --out datasets/buildings_100_v1/metrics_levels.json
```

`--pred_source rebuild_world` がデフォルトです。  
再建築結果が `building_xxx/rebuild_world/{voxels.npy,bbox.json}` にあればそれを優先し、`--allow_gt_fallback` を付けると `building_xxx/gt` にフォールバックできます。

## LLM APIキー設定
`GPT` と `Claude` のキーは `.env` に置けます。

1. テンプレートをコピー:
```bash
cp .env.example .env
```

2. `.env` を編集してキーを設定:
- `OPENAI_API_KEY=...`
- `ANTHROPIC_API_KEY=...`
- `LLM_PROVIDER=openai` または `anthropic`
- 低コスト寄りの推奨モデル:
  - `OPENAI_MODEL=gpt-5-mini`
  - `ANTHROPIC_MODEL=claude-haiku-4-5-20251001`

3. 読み込み確認:
```bash
python3 tools/llm_config.py
```

## image-to-text-to-build 実験
追加したCLI:

- `tools/generate_building_descriptions.py`
  - 画像 -> 説明文(JSON)
- `tools/generate_rebuild_plans.py`
  - 説明文 -> 再建築計画(JSONアクション列)
- `tools/render_rebuild_from_plan.py`
  - 計画 -> `rebuild_world/{voxels.npy,bbox.json}`
- `tools/evaluate_description_quality.py`
  - 説明文の自動評価
- `tools/evaluate_rebuild_metrics.py`
  - 再建築の一致度評価（Level 0-4）
- `tools/repair_rebuild_plans.py`
  - 既存 `rebuild_plan` を新しいスキーマ拘束・材質整合ロジックで再修復（LLM再呼び出し不要）
- `tools/self_refine_rebuild_plans_no_gt.py`
  - planを一度レンダリングして自己整合スコアを計算し、GTなしで反復補正（post-render self-refine）
- `tools/run_i2t2b_experiment.py`
  - 上記を一括実行
- `scripts/run_i2t2b_experiment.sh`
  - 一括実行ラッパー

### rebuild_plan のスキーマ自動修復（重要）
`tools/generate_rebuild_plans.py` は、LLM出力の表記ゆれを吸収して `fallback` を減らすために以下を自動修復します。

- operation配列キー揺れ: `operations`, `ops`, `actions`, `steps`, `commands`
- op種別揺れ: `op/type/action/kind/cmd` + `setblock`, `clear`, `replace`, `outline`, `hollow_box`, `windows_row` など
- 座標揺れ:
  - `x1..z2`
  - `from/to`, `pos1/pos2`, `min/max`, `start/end`
  - `bbox/box/cube` 配列 (`[x1,y1,z1,x2,y2,z2]`)
- blockキー揺れ: `block`, `material`, `replace_with`, `outer_block`, `block_glass` など
- critic-revise が壊れた場合の救済:
  - revised が空操作列なら draft を再採用
  - それでも空ならヒューリスティックへフォールバック

`plan.json` には `coerce_report` が保存され、`plan.request.json` にも `parse_report/coerce_report` が残るため、どこで修復されたか追跡できます。

### rebuild_plan の材質制約強化（role固定 + budget + strict schema）
`prompts/rebuild_plan_strict_material_v3.json` を使うと以下を有効化できます。

- role固定:
  - 非`carve`操作は `role` を推定/補完し、`block == palette[role]` へ自動修復
- material budget:
  - `material_budget`（wall/roof/trim/glass/light/floor）を必須化
  - `target_blocks` と実測ブロック量の乖離を `validation_report.budget_violations` へ記録
- strict schema:
  - `bbox/operations/palette/material_budget/self_check` の整合を検証し `validation_report` を保存
  - `max_operations` を超えた場合は切り詰めて `issues` に記録

実行例:
```bash
scripts/run_i2t2b_experiment.sh \
  --dataset_root datasets/buildings_100_v1 \
  --provider openai \
  --dotenv .env \
  --plan_prompt_profile prompts/rebuild_plan_strict_material_v3.json \
  --plan_critic_revise
```

`plan.json` と `plan.request.json` の両方に `validation_report` が保存されます。

### 既存planを後段だけ強化して再評価（v5）
LLM再実行せず、既存の `rebuild_plan_*` を後処理ロジックで修復できます。

```bash
python3 tools/repair_rebuild_plans.py \
  --dataset_root datasets/buildings_100_v1 \
  --source_plan_subdir rebuild_plan_pe_v2_openai_gpt_5_mini \
  --out_plan_subdir rebuild_plan_schema_material_v5_repair_openai_gpt_5_mini \
  --description_subdir description_openai_gpt_5_mini \
  --strict_schema --enforce_role_fixed --require_material_budget \
  --prefer_description_palette \
  --material_budget_tolerance 0.35 \
  --role_fix_min_confidence 0.78
```

その後:
```bash
python3 tools/render_rebuild_from_plan.py \
  --dataset_root datasets/buildings_100_v1 \
  --plan_subdir rebuild_plan_schema_material_v5_repair_openai_gpt_5_mini \
  --out_subdir rebuild_world_schema_material_v5_repair_openai_gpt_5_mini \
  --overwrite

python3 tools/evaluate_rebuild_metrics.py \
  --gt_root datasets/buildings_100_v1 \
  --pred_root datasets/buildings_100_v1 \
  --pred_subdir rebuild_world_schema_material_v5_repair_openai_gpt_5_mini \
  --out datasets/buildings_100_v1/metrics_levels_schema_material_v5_repair_openai_gpt_5_mini.json
```

v5ロジックの主変更:
- `description.materials` 由来で palette を補正（role別）
- operation の role 推定を幾何特徴（高さ・境界接触・薄板判定）込みで実施
- `block == palette[role]` の強制は `role_fix_min_confidence` 以上の高信頼操作に限定
- `self-repair pass` を追加（plan -> renderer 前）:
  - 欠落要素（floor/wall/roof/window/entrance/light）を最小追加
  - `window` 材質ミスマッチの自動補正
  - `coarse -> decorative` の2段並び替えを強制
- 高レベルopの内部展開に対応:
  - `roof_template`, `window_pattern`, `slope`
  - 内部で `fill/carve/set` に展開して実行

### post-render自己整合補正（GTなし）
`tools/self_refine_rebuild_plans_no_gt.py` は、既存 plan を仮想レンダリングして自己整合スコアを計算し、スコアが改善する場合のみ補正opを採用します。

- 評価成分（GT不要）:
  - 材質比率整合（material budget / 推定比率）
  - 形状寸法整合（descriptionの `dimensions_estimate`）
  - 屋根整合（上層のroof比率 + taper）
  - 窓/入口の存在整合
- 補正内容:
  - `roof_template` / `window_pattern` / 入口 carve / role別不足分の補強
  - 補正後に再検証し、`min_score_gain` 未満なら不採用
  - render後に材質budget再投影（不足role追加 + 過剰role削減）を実行可能
  - 屋根/窓は単一案ではなく複数テンプレ候補を探索して最良案を採用

単体実行例:
```bash
python3 tools/self_refine_rebuild_plans_no_gt.py \
  --dataset_root datasets/buildings_100_v1 \
  --plan_subdir rebuild_plan_schema_material_v5_repair_openai_gpt_5_mini \
  --description_subdir description_openai_gpt_5_mini \
  --out_plan_subdir rebuild_plan_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt \
  --max_iterations 2 \
  --min_score_gain 0.01 \
  --max_added_ops_per_iter 64 \
  --roof_search_variants 8 \
  --window_search_variants 8 \
  --max_search_candidates 20 \
  --enable_material_budget_reprojection \
  --material_budget_reprojection_strength 0.6 \
  --material_budget_reprojection_min_deficit_ratio 0.025 \
  --material_budget_reprojection_trigger_material_score 0.78 \
  --selection_op_penalty 0.001
```

一括パイプラインで有効化する場合:
```bash
scripts/run_i2t2b_experiment.sh \
  --dataset_root datasets/buildings_100_v1 \
  --provider openai \
  --dotenv .env \
  --description_subdir description_openai_gpt_5_mini \
  --plan_subdir rebuild_plan_schema_material_v5_repair_openai_gpt_5_mini \
  --enable_self_refine_no_gt \
  --self_refine_plan_subdir rebuild_plan_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt \
  --self_refine_max_iterations 2 \
  --self_refine_min_score_gain 0.01
```

モデル別最適設定（tuned, 200件フル）:
- OpenAI（弱め再投影）: `strength=0.4`, `selection_op_penalty=0.0015`
- Claude（強め再投影）: `strength=0.6`, `selection_op_penalty=0.001`

実行例（OpenAI tuned plan -> render -> eval）:
```bash
python3 tools/self_refine_rebuild_plans_no_gt.py \
  --dataset_root datasets/buildings_100_v1 \
  --plan_subdir rebuild_plan_schema_material_v5_repair_openai_gpt_5_mini \
  --description_subdir description_openai_gpt_5_mini \
  --out_plan_subdir rebuild_plan_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned \
  --roof_search_variants 8 \
  --window_search_variants 8 \
  --max_search_candidates 20 \
  --material_budget_reprojection_strength 0.4 \
  --selection_op_penalty 0.0015
python3 tools/render_rebuild_from_plan.py \
  --dataset_root datasets/buildings_100_v1 \
  --plan_subdir rebuild_plan_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned \
  --out_subdir rebuild_world_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned
python3 tools/evaluate_rebuild_metrics.py \
  --gt_root datasets/buildings_100_v1 \
  --pred_root datasets/buildings_100_v1 \
  --pred_subdir rebuild_world_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned \
  --out datasets/buildings_100_v1/metrics_levels_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned.json
```

limit=10 の実測（buildings_100_v1, v5_repair系）:

- OpenAI:
  - baseline: IoU `0.2970`, F1 `0.4406`, material_match `0.2193`
  - self_refine_no_gt: IoU `0.3344`, F1 `0.4903`, material_match `0.3967`
- Claude:
  - baseline: IoU `0.2289`, F1 `0.3697`, material_match `0.1567`
  - self_refine_no_gt: IoU `0.2526`, F1 `0.4000`, material_match `0.1884`

評価JSON:
- `datasets/buildings_100_v1/metrics_levels_schema_material_v5_repair_openai_gpt_5_mini_l10_baseline_compare.json`
- `datasets/buildings_100_v1/metrics_levels_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_l10.json`
- `datasets/buildings_100_v1/metrics_levels_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_l10_baseline_compare.json`
- `datasets/buildings_100_v1/metrics_levels_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_l10.json`

### 一括実行（推奨）
```bash
scripts/run_i2t2b_experiment.sh \
  --dataset_root datasets/buildings_100_v1 \
  --provider openai \
  --dotenv .env \
  --thresholds_json tools/thresholds_levels.example.json
```

デフォルトで `provider + model` タグにより出力先が自動分離されます（上書き回避）。
例: `description_openai_gpt_5_mini`, `rebuild_plan_openai_gpt_5_mini`, `rebuild_world_openai_gpt_5_mini`

`--overwrite` を付けない限り既存結果は上書きしません。

### 小規模テスト（5件）
```bash
scripts/run_i2t2b_experiment.sh \
  --dataset_root datasets/buildings_100_v1 \
  --provider openai \
  --dotenv .env \
  --limit 5
```

明示タグを使う場合:
```bash
scripts/run_i2t2b_experiment.sh \
  --dataset_root datasets/buildings_100_v1 \
  --provider anthropic \
  --dotenv .env \
  --output_tag anthropic_haiku_retry1
```

### 主要出力
- `building_xxx/description/description.json`
- `building_xxx/rebuild_plan/plan.json`
- `building_xxx/rebuild_world/voxels.npy`
- `building_xxx/rebuild_world/bbox.json`
- `datasets/.../description_metrics.json`
- `datasets/.../metrics_levels.json`


buildings_100_v1 / openai_gpt_5_mini
auto_score_mean=0.8102, IoU=0.2872, F1=0.4418, material_match=0.2679
buildings_100_v1 / anthropic_claude_haiku_4_5_20251001
auto_score_mean=0.7202, IoU=0.3025, F1=0.4610, material_match=0.2042
buildings_100_v4 / openai_gpt_5_mini
auto_score_mean=0.7520, IoU=0.1522, F1=0.2604, material_match=0.3246
buildings_100_v4 / anthropic_claude_haiku_4_5_20251001
auto_score_mean=0.6893, IoU=0.1205, F1=0.2130, material_match=0.2893

## 実験結果サマリ
- まず最初に見る（結論だけ）: `reports/final_results_concise_ja.md`
- 最終報告（本実験全体）: `reports/final_experiment_report_ja.md`
- 最新の考察・比較まとめ: `reports/experiment_summary_2026-03-01.md`
- 統計的検証 + アブレーション + 外的妥当性（CI/p値付き）: `reports/statistical_validity_ablation_external_validity_ja.md`
- 再建築評価（IoU/F1等）の詳説: `reports/rebuild_metrics_guide_ja.md`
- 2種類実験（baseline vs 強化プロンプト）の比較: `reports/two_experiment_types_summary_ja.md`
- 統計JSON集計: `reports/statistics/stat_ablation_external_summary_2026-03-01.json`
- 検定のseed反復安定性（resampling）: `reports/statistics/seed_repeat_resampling/summary.md`
- 低IoU失敗ケースの系統分析（件数 + 修正優先順位）: `reports/failure_analysis/low_iou_failure_taxonomy_2026-03-01.md`
- fallback削減（schema拘束 + parser_v6）の図付きレポート: `reports/fallback_reduction_parser_v6_report_ja.md`
- モデル別外的妥当性比較:
  - `reports/statistics/external_validity_openai_gpt_5_mini.md`
  - `reports/statistics/external_validity_anthropic_claude_haiku_4_5_20251001.md`
- 図を再生成する場合: `python3 tools/plot_experiment_figures.py`
- 2種類比較図を再生成する場合: `python3 tools/plot_experiment_type_comparison.py`
- self_refine比較図を再生成する場合: `python3 tools/plot_self_refine_comparison.py`
- parser_v6 fallback比較図を再生成する場合: `python3 tools/plot_fallback_reduction_parser_v6.py`
- 日本語ラベル版図は `reports/figures/*_ja.svg` に出力
