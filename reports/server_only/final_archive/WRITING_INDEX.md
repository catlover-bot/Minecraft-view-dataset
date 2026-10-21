# 執筆用成果物索引

この索引は、サーバー終了前に既存成果を読み取り・執筆用へ退避するためのもの。新しい論文本文、実験、採点、統計検定、CI、画像レンダリングは作成していない。過去のREADMEはその作成時点の状態を記録しているため、後日の実行結果と分けて読む。生成runner、launcher、保存済みauthorityは復元後も起動せず、分析・閲覧専用資料として扱う。

paper-assets内ではrepository配下の相対配置を維持する。以下の相対リンクは元repositoryでの配置であり、別の絶対pathを持つcanonical成果は最終復元セットの`ORIGINAL_PATH_MAP`から解決する。元サーバーの同名ファイルへのfallbackを前提にしない。実際に退避されたmember・SHA・取得先は最終coverageと復元receiptが根拠であり、この索引への列挙だけではGitHub保存済みを意味しない。

| 執筆上の用途 | 優先する既存成果物 | 範囲・注意 |
|---|---|---|
| 主実験と失敗分母 | [Goal007 README](../goal007/README.md)、[960行CSV](../goal007/data/final_slots_960.csv)、[960行JSON](../goal007/data/final_slots_960.json)、[source mapping](../goal007/data/source_to_slot_mapping.csv)、[集計](../goal007/results/results.json) | 60建築・960予定枠。934成功、25既知モデル出力失敗、運用欠測1。正常empty successを失敗へ変更しない。991 historical callsは独立建築Nではない。 |
| 主表・比較・図の元データ | [per_group](../goal007/results/per_group.csv)、[20比較](../goal007/results/comparisons_20.csv)、[分母](../goal007/results/denominators.json)、[figure sources](../goal007/results/figure_sources.json)、[日本語所見](../goal007/results/SUMMARY_JA.md) | 保存された検定family・欠測規則をそのまま使う。後の探索結果で置換しない。 |
| 既存TeX・図版・執筆断片 | [Goal007 main table](../goal007/results/table_main.tex)、[統計表](../goal007/results/table_statistics.tex)、[ablation](../goal007/results/table_ablation.tex)、[失敗表](../goal007/results/table_failures.tex)、[Goal008 publication](../goal008/publication/)、[既存manuscript sections](../goal008/publication/manuscript_sections.tex) | 既存の執筆素材。ここで新しいmanuscriptを執筆したものではない。PNG/SVGとTeX参照先も同じ配置で保存する。 |
| GT属性proxyと事例図 | [Goal008 README](../goal008/README.md)、[180 proxy](../goal008/phase1/proxy_scores_180.csv)、[360探索](../goal008/phase1/exploratory_associations_360.csv)、[source map](../goal008/phase1/source_map_180.csv)、[事例図](../goal008/examples/figures/) | GT由来属性proxyは独立caption品質ではない。従来RQ2の代替検証として扱わない。 |
| 箱baseline・要因・Exact-GT・頑健性 | [Goal010 README](../goal010/README.md)、[baseline表](../goal010/analysis/phase1_baselines/baseline_per_building.csv)、[対応差](../goal010/analysis/phase1_baselines/baseline_matched_deltas.csv)、[provider対比](../goal010/analysis/phase2_factors/description_builder_matched.csv)、[E容量](../goal010/analysis/phase3_exact_oracle/E_capacity_diagnostics.csv) | 保存済み480 baselineと元予測の対照。P/Eは追加GT情報を含む。visibilityは未検証という元の制約を維持する。 |
| 占有・表面・材質を分ける指標 | [Goal010v2 README](../goal010v2/README.md)、[960行追加指標](../goal010v2/analysis/3_full/new_metric_values_960.csv)、[分母](../goal010v2/analysis/4_summary/metric_denominators.csv)、[baseline対比](../goal010v2/analysis/4_summary/metric_baselines_paired.csv)、[E診断図](../goal010v2/analysis/4_summary/E_capacity_diagnostic.svg) | post-hoc CPU診断。元occupancy shiftを共有する各指標は異なる側面を測る。surfaceを可視外観・機能へ一般化しない。元JSONの公開gzipも保持する。 |
| E材質・研究判断・先行研究 | [Goal011](../goal011/README.md)、[101 E事例](../goal011/E_CASES_101.csv)、[mapping上界](../goal011/E_MAPPING_UPPER_BOUNDS.csv)、[Goal012](../goal012/README.md)、[RQ証拠gate](../goal012/RQ_EVIDENCE_GATES.md)、[先行研究行列](../goal012/PRIOR_ART_COLLISION_MATRIX.md) | GT-informed label対応上界は実モデル成功へ加算しない。元RQの未検証部分と既存文献調査の時点を維持する。新しい実験提案はこの退避では行わない。 |
| CLIPScore summary整合度 | [Goal013 source180](../goal013/SOURCE_BINDING_180.json)、[Goal013A README](../goal013a/README.md)、[score180 CSV](../goal013a/SCORES_180.csv)、[score180 JSON](../goal013a/SCORES_180.json)、[token180](../goal013a/TOKEN_COVERAGE_180.csv)、[crop360](../goal013a/CROP_METADATA_360.json)、[相関](../goal013a/POOLED_SPEARMAN.csv)、[建築内一致](../goal013a/WITHIN_BUILDING_CONCORDANCE.csv) | 180予定行・178有効summary・1068画像対。2空summaryは欠測。公式ViT-B/32の単画像qを6枚平均する独自集約。人間品質妥当性・因果性は未検証。今回再採点していない。 |
| CLIP対応建築・非対応対照 | [Goal014 README](../goal014/README.md)、[QA180](../goal014/QA_ROWS_180.csv)、[12推定](../goal014/ESTIMATES_12.csv)、[判断](../goal014/RESEARCH_DECISION_JA.md) | 178×60行列・保存特徴・固定対照からの局所診断。元CLIP得点や主実験を置換しない。特徴cacheとfull CI drawは最終復元セット側も保持する。 |
| 未完・退化したS/F | [Goal015仕様](../goal015/README.md)、[Goal015A親保存証明](../goal015a/PARENT_SAVE_AUDIT.json)、[Goal015B所見](../goal015b/README.md)、[80行raw監査](../goal015b/RAW_RESPONSE_AUDIT_80.csv)、[48empty分類](../goal015b/EMPTY_CLASSIFICATION_48.csv)、[打切り・検証失敗](../goal015b/TRUNCATION_VALIDATION_AUDIT.csv) | 80予定枠中72送信・8NOT_SENT。72内訳は48正常empty、22truncation、2validation、非empty0。残8件を完了にしない。S/F差の退化を同等性や全文情報の無効性と解釈しない。 |
| schema順序：合成課題C | [12行CSV](../goal015d/MINI_RESULTS_12.csv)、[正式集計](../goal015d/MINI_RESULTS_SUMMARY.json)、[解釈](../goal015d/MINI_INTERPRETATION_JA.md)、[Goal015C仕様](../goal015c/README.md) | 合成課題1種類。OpenAI A0/2 B2/2 C2/2、Anthropic A0/2 B2/2 C0/2。task_passとprotocol-valid・3操作列適合は別。C/Anthropicのfence失敗を修復して成功へ加算しない。 |
| schema順序：実建築D | [今回転記した実測16行](GOAL015D_RESULTS_16.csv)、[日本語要約](GOAL015D_RESULTS_SUMMARY_JA.md)、[固定した転記元](GOAL015D_RESULTS_SOURCE.json)、[元準備仕様](../goal015d/EXPERIMENT_SPEC.md) | 実建築2件のdevelopment canary。A非空0/8、B8/8。Aの正常empty5件とtruncation3件を分離。実費USD0.67051700。準備READMEだけを最終実績として引用しない。 |

## 復元後に開く最小の正式row表

以下は別の研究scopeであり、行数を合算して独立Nにしない。JSONの保存値を開くだけでよく、runnerや評価処理を実行する必要はない。

| 検査対象 | original_path | JSON内の行位置 |
|---|---|---|
| 主実験960行 | `/root/workspace/Minecraft-view-dataset/reports/server_only/goal007/data/final_slots_960.json` | top-level array、960 |
| CLIP180予定行 | `/root/workspace/Minecraft-view-dataset/reports/server_only/goal013a/SCORES_180.json` | top-level array、180 |
| C12行 | `/root/minecraft-runs/goal015c-20261020T094646Z-fcb2ec03/evaluations/goal015c-20261020T105050090371Z-22988535/evaluation_12.json` | `rows`、12 |
| D16行 | `/root/minecraft-runs/goal015d-20261020T114450Z-ebe767ac/evaluations/goal015d-20261020T125031610548Z-d17a0619/evaluation_16.json` | `rows`、16 |
| 親S/F80予定枠 | `/root/minecraft-runs/goal015-20261017T170546Z/evaluations/goal015a-parent72-20261019T141842Z-13cfbc79/evaluation_80.json` | `rows`、80。`preparation-check`版を実測として使わない |

主960行の`artifacts`にはraw・parsed・voxelsの相対pathがある。対応する`source_run`／`source_label`と[Goal007 MANIFEST](../goal007/MANIFEST.json)のsource mappingから原本を解決する。先行160件はGoal005の別sourceで、継続781試行と補完3・17の保存だけでは網羅しない。既存先行160保存はRelease412585766／asset640015775。個別Releaseの参照だけで完全復元を保証せず、最終coverageが外部依存を含めて束縛する。

Goal013 source180は原description/request、scene camera/meta、順序付き入力画像のpath/hashを持つ。GTは各実験planの`gt_sources`やdata source manifest、予測は選択済みslotの`artifacts`を使う。生応答や入力の原文は新しいpublic Git資料へ貼り付けず、最終非公開Draftのデータmemberとして扱う。

## 初期200建築・旧1200出力・既存執筆断片

初期の`buildings_100_v1`と`buildings_100_v4`合計200建築、および既存1200出力は、後の60建築・960枠とは版・条件が異なる。固有原本と既存診断は保全し、比較可能性を追加で主張しない。

- [初期成果索引](../../final/artifacts_index.md)、[200建築の旧まとめ](../../final/summary_ja.md)、[共通ハイパラの既存表](../../final/main_shared_hparams_results_2026-03-10.md)、[JSON](../../final/main_shared_hparams_results_2026-03-10.json)、`reports/figures/`、`reports/statistics/`、`reports/failure_analysis/`、`reports/support/`を保持する。
- [Goal002 README](../goal002/README.md)と[1200出力のstrict alignment比較](../goal002/alignment_comparison.csv)、[legacy集計](../goal002/legacy_geometry_summary.json)、[材質診断](../goal002/material_diagnostics.json)は、geometry1200・material1162・UNKNOWN38を記録する。historical ARR原本不足と、保存済みlegacy i2t2b v5の再評価を混同しない。
- 旧成果索引には、OpenAIの常時diversify設定のrawと40ディレクトリが以前に削除済みと記されている。これは`PREEXISTING_MISSING`として扱い、今回の退避で消えたファイルや再生成済みデータとはしない。その他の欠落は最終coverageに従う。
- `offline_evaluation/experiments/arr_followup_v1/paper_tables/*.tex`と`paper_text/limitations.tex`は既存の断片として保持する。ファイルがあること自体を新しい実験・人手評価の実施証明にしない。`reports/final/`のhuman用templateも実測結果ではない。
- repository内の通常ファイル探索では独立した`.bib`は見つからなかった。存在しないbibや全文manuscriptを新規作成して補わず、既存TeX・Markdownの出典とGoal012の文献資料を保持する。フォントは同梱対象外。

## 再現環境と保存状態の読み方

一般の評価は既存Python3.12環境、CLIPは専用Goal013A環境を使った。環境版・lock・ソースcommit・checkpointの公式取得先とSHA・利用条件は[Goal013A環境記録](../goal013a/ENVIRONMENT_RECEIPT.json)、[依存manifest](../goal013a/DEPENDENCY_MANIFEST.json)、[lock](../goal013a/requirements.lock)、各run/specに残る。venv・pip cache・OS・フォントを丸ごと同梱しない。再取得可能な公開ソフトと、唯一の入力/raw/feature/scoreを別扱いにする。

原本の絶対pathや署名は書き換えず、復元マッピングを別保存する。GitHubからの取得と全member照合、PCへのコピー状態は最終保存案内・coverage・復元receiptを参照する。同一サーバー上の空directoryでの復元確認は、利用者PCでの確認とは異なる。PC確認がなければ`AWAITING_USER_COPY`のままとする。この退避作業でサーバー停止・解約・削除は行わない。
