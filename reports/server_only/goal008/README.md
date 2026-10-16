# Goal008：探索的属性proxy・再建築例・ARR論文用成果物

現行の60建築・180記述を使い、新しい **exploratory CPU-derived attribute fidelity proxy** を計算した。従来の説明品質スコアではなく、GTとの材質・寸法・フィールド充足度の一致を測る。元のRQ2は **NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES** のまま。研究API送信・モデル照会・token-count API・キー照会・人手実験はすべて0回。

## 表現条件と出典

Oは保存済みv4のJSON形式の視覚的説明レコードそのもの。summary、materials、dimensions、elements、hintsなどの人間可読フィールドを含む。Jは同じtyped JSON treeを保つ再シリアライズ。Pは一部GT属性、Eは全離散GT仕様を与える診断条件で、純粋な言語情報損失の因果介入ではない。

説明生成はOpenAI・Anthropic・Geminiの3 provider、再建築BuilderはOpenAI・Anthropicの2 providerで、役割を分けた。180/180記述で原文SHA256・input ID/hash・GT bbox/voxel/grid hashを一致確認した。[表現監査](audit/phase0_semantics.md)、[原文対応表](phase1/source_map_180.csv)、[sampling監査](audit/sampling_audit.json)に証跡を残した。

O/Jは各360 slotに対する別の生成呼出しで、attemptはO=361、J=360。追加O attemptはglobal940の承認済みretryであり、科学的な反復測定ではない。requestはtemperature・top-p・decoding seed等を明示固定していない。保存されたOpenAI response metadataはtemperature=1、top-p=1を報告するが、Anthropicの該当値は不明。J−O差を表記変更だけの因果効果とは断定できない。

## 新しい探索的proxyと結果

固定重みはstrict material F1 / coarse material F1 / dimension relative accuracy / typed field completeness = **0.30 / 0.25 / 0.25 / 0.20**。欠損・不正寸法をGTで補完しない。曖昧材質は事前指定した不一致トークンとして扱い、strict材質名には同義語置換を加えない。[算出仕様](definition.md)と[統計仕様](stats_definition.md)を実スコア計算前にcommit・pushして固定した。

180記述で寸法の欠損・型不正は0件、曖昧・不正材質トークンを含む記述は10件。completenessは178件で1、2件で0.2となり、変動が乏しい。各成分と合成値を[180件JSON](phase1/proxy_scores_180.json) / [CSV](phase1/proxy_scores_180.csv)へ保存した。JSONの継承属性 `score_status=NOT_COMPUTED_BY_SOURCE_AUDIT` はPhase0時点を表し、新規計算値は `scores` と `scoring_specification_id` に格納されている。

合成proxyとO aligned IoUの対応は次のとおり。CIは60建築をクラスタとする2,000回bootstrap、seed20261016。Builderの2観測を独立N=360とは数えない。

| Builder対象 | valid記述 / 建築 | Spearman [95% CI] | 建築内中心化Pearson [95% CI] |
|---|---:|---:|---:|
| OpenAI | 177 / 60 | 0.3932 [0.2514, 0.5217] | 0.4033 [0.2204, 0.6077] |
| Anthropic | 179 / 60 | 0.3745 [0.2278, 0.5126] | 0.4042 [0.2409, 0.5761] |
| 両Builderがvalidな平均 | 176 / 60 | 0.4183 [0.2662, 0.5488] | 0.4478 [0.2607, 0.6358] |

建築内midrank偏差の関連は順に0.3939 [0.2294, 0.5528]、0.3502 [0.1667, 0.5190]、0.4035 [0.2444, 0.5612]。既知O出力失敗3件/1件だけを0にする別のbounded-IoU感度分析では、各180記述でSpearmanが0.4188 / 0.3736 / 0.4328となった。運用上の未知観測を0にしたものではない。

5成分×4 provider範囲×2 outcome policy×3 Builder対象×3関連の **360分析をすべて保存**。144件は点推定とCI、24件は点推定のみ（completenessの有限bootstrapが1,747/2,000で事前閾値1,900に届かない）、12件はproxy定数、180件はprovider別で1記述/建築のため建築内分析不能。都合のよい指標だけを選んでいない。[全360件](phase1/exploratory_associations_360.csv)、[詳細要約](EXPLORATORY_RQ2_SUMMARY.md)、[独立検算](independent_phase1_review.json)を参照。

これらは同じGTを参照する属性proxyと再建築結果の探索的関連であり、人間による説明品質判断の妥当性、従来caption qualityの予測性能、因果性を証明しない。

## 元の結果を変更していないこと

960固定slot、959 selected completed、934 valid、25モデル出力失敗（truncation19・validation6）、1運用欠測（global942 HTTP520）を保持した。942のusage/response/metricsはnullのまま。global940の元401と承認済み同一payload成功retryを別attemptとして保持し、slotは1つと数えた。固定scopeの961 attempts、歴史的991 calls、未知usage予約2件、予算計上USD21.56083625も不変。予算計上は実請求額ではない。

元の20比較・8/4/8 family・bootstrap2,000・sign-flip10,000・family内Holmは不変で、探索的360分析を追加していない。5,811原本ファイル、Goal007関連161ファイル、凍結実装57ファイルを照合した。旧ARR200件・過去1200件・v5と混ぜていない。

Goal007 MANIFEST SHA256は `ef05c2ed93247802e9a68a00d700d39a8934d6231c7787315ab37b01d2467714`。4 source系統・plan/ledger/authority/checkpointの参照は[MANIFEST](MANIFEST.json)に保持する。既存Draft Release 414157245 / 414396750 / 414429861、各6 assetsの復元済み証拠を再利用した。再ダウンロード・Draft公開・既存asset変更は0回。

## 実データ再建築例

BuilderとO/J説明providerをOpenAIに固定し、O/J/P validの55候補から、[事前固定した規則](example_definition.md)で異なる3建築を選んだ。同点はscene ID辞書順、置換なし。proxy結果を選択に使っていない。

| scene（v4） | 選択規則 | O IoU | O→P寸法MAE | E |
|---|---|---:|---:|---|
| building_034 | 候補O IoU中央値に最も近い | 0.1986 | 15.0000→3.3333 | valid |
| building_088 | 残る候補の最小O IoU | 0.0864 | 13.0000→8.3333 | truncation |
| building_024 | P IoU≤0.20かつ正の寸法改善が最大 | 0.1169 | 21.0000→0.0000 | truncation |

GT/O/J/P/Eを同じabsolute XYZ・カメラ・物理スケール・共通材質色で描画した。選択には保存済みstrict-aligned指標を使うが、図にはalignment shiftを適用していない。元voxelの露出面だけを描き、E失敗2件には形状を作らずtruncationと表示した。具体例から母集団平均を一般化しない。

- [例1 PNG](examples/figures/example_01.png) / [SVG](examples/figures/example_01.svg)、[例2 PNG](examples/figures/example_02.png) / [SVG](examples/figures/example_02.svg)、[例3 PNG](examples/figures/example_03.png) / [SVG](examples/figures/example_03.svg)
- [選択候補・除外理由](examples/figures/example_selection.json)、[原文抜粋・GT bbox・15 panelのattempt/input/source hash](examples/figures/example_sources.json)、[指標と出典対応表](examples/figures/example_companion.csv)

初回の図は形状・出典に問題がなかったが、論文幅では文字が小さかったため、文字・凡例・注記だけを修正した。初回runは上書きせず保存し、選択・voxel・指標は維持した。[初回検証](examples_initial_visual_source_review.json)と[初回manifest](initial_examples_manifest.json)を残している。

## Overleaf用成果物

[publication](publication/)のTeXとPNGを同じOverleafディレクトリへ配置する。必要パッケージはgraphicxとbooktabs。

- 表：[main_results.tex](publication/main_results.tex)、[ablations.tex](publication/ablations.tex)、[failures.tex](publication/failures.tex)、[3 family別stats_appendix.tex](publication/stats_appendix.tex)、[材質・repair誤差](publication/error_structure.tex)、[全探索分析付録](publication/exploratory_appendix.tex)
- 図：[主要誤差 PNG](publication/figure_main_errors.png) / [SVG](publication/figure_main_errors.svg)、[proxy関連 PNG](publication/figure_proxy_scatter.png) / [SVG](publication/figure_proxy_scatter.svg)、3実例を1図にまとめる[examples_figure.tex](publication/examples_figure.tex)、[キャプション](publication/figure_captions.tex)
- 英文：[Results](publication/Results.md)、[Discussion](publication/Discussion.md)、[Limitations](publication/Limitations.md)、[TeX本文](publication/manuscript_sections.tex)、[事後的RQ再構成の明示](publication/RQ_REFRAMING.md)
- 数値出典：[publication_numbers.json](publication/publication_numbers.json)、[描画点176件](publication/proxy_scatter_points.json)、[RUN manifest](publication/RUN.json)

元のGoal007図表は上書きしていない。既存TeXコンパイラがないためPDFコンパイルは **NOT_TESTED_COMPILER_UNAVAILABLE**。依存取得・OS変更は行っていない。数値照合、TeX静的検査、PNG目視はコンパイル成功と区別する。

## 投稿前に残る限界とAPIなしの次作業

最重要の限界は、新proxyが妥当性検証済みの人間品質指標ではないこと、GTを共有する関連であること、最大60建築の固定cohortと単発生成に依存すること。valid選択、Eのtruncation、運用欠測、探索的多重性も残る。O/J差を表記だけの因果効果とせず、Eの異なるvalid分母による単群平均でBuilder優劣を主張しない。repairは操作数で、人間の作業時間ではない。

追加APIなしの最短の次作業は、この図表と英文をARRテンプレートへ配置し、既存のOverleaf環境でPDFをコンパイルして改ページ・表幅を確認し、主張と表・図・分母の対応を最終点検すること。従来RQ2は未検証として残す。

## CPU検証・環境・保存

最新の固有テストは **131 PASS / 0 FAIL / 0 ERROR / 0 SKIP**。内訳はPhase0/1の74件、実例35件、論文出力22件。同じテストの再実行や過去Goalの件数は加えていない。[Phase0/1 JUnit](phase1_tests_final.junit.xml)、[実例テスト](examples_publication_fixture_tests.json)、[論文出力JUnit](publication_fixture_tests_final_bound.junit.xml)を参照。

初期のscorer import collection error、統計fixtureの4 assertion failures、matplotlibのない環境での出版fixture失敗は、修正前のログを残した。既存の描画用venvへ切り替え、新規依存取得は行っていない。実データPhase0/1・初回実例・最終実例・出版処理はいずれも終了0。

| 実処理 | 経過秒 | CPU秒（user+system） | 最大RSS KiB |
|---|---:|---:|---:|
| Phase0原本・意味監査 | 18.28 | 18.26 | 122468 |
| Phase1 proxy・関連解析 | 25.72 | 25.70 | 147388 |
| 最終実例描画 | 27.97 | 27.93 | 232676 |
| 出版出力＋前後原本検証 | 11.65 | 11.62 | 154864 |

この4処理の合計経過は83.62秒、CPUは83.51秒。初回実例描画31.28秒・最大RSS273296 KiBも別に実行し、上書きせず保存した。テスト・独立検算・Git通信の時間はこの合計に含めない。出版renderer単体の内部計測は2.291秒。実行別の`*.resources.txt`とmanifestを保存している。最大RSSは合算しない。最終実例出力時の空き容量は69,532,229,632 bytes。

実行環境はroot、`/root/workspace/Minecraft-view-dataset`、Python3.12.3、workspace-write / auto_review。通常sandboxコマンドが成功し、Git保存とrunディレクトリへの書込みだけ対象別承認を使った。設定値からOSレベルの隔離を保証していない。[環境証跡](environment.json)を参照。BLAS/OpenMP/MKL=1。CPU用と描画用の既存専用venvを使用し、root接続・OS・SSH・既存venv・チェックポイントを変更していない。

開始HEADは `c75efa28f4871ce2f4263bf8921041c4c162ede9`。Phase1凍結commitは `6a5b2b99f4566bbd7dde5dbc7c44c866b19fe802`、最終実例・出版コードを含むpush済みcommitは `7bd5183cdcab7506c25d163f74d45075cdec700c`。branchは `research/server-only-20261013`、mainは `7309327efd1c19dda793b19e8d8e6ce0b5868db9` のまま。[コード保存証跡](final_render_code_push_receipt.json)を参照。

新規変更は `offline_evaluation/analysis/goal008/` と `reports/server_only/goal008/` に限定。元応答・voxel・認証情報を公開成果物に複製していない。全新規ファイルを公開安全scanし、今回の運用上限（単一10MiB）を確認する。新しいDraft Releaseは不要。最終報告commit・remote一致・main不変・worktree cleanの照合は、同run rootの新規 `publication_receipt.json` と完了報告に記録する。

原本run rootは `/root/minecraft-runs/goal008-analysis-20261016T151245Z`。`audit`、`phase1`、`examples-publication-size`、`publication`が最終出力、`examples`が保全した初回実例。公開コピーは[Phase0/1照合](phase01_public_copy.json)と[最終実例・出版照合](examples_publication_public_copy.json)でbyte一致を確認した。

## CPU再実行

repo rootから実行する。既存runを上書きせず、新しい一意の`GOAL008_REPLAY`を選ぶ。workspace外への書込みには対象別承認が必要。原本や凍結コードが不一致なら実行器は停止する。これらのCLIは研究APIを呼ばない。

```bash
set -e
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
GOAL008_REPLAY=/root/minecraft-runs/goal008-replay-CHOOSE_NEW_UTC
mkdir "$GOAL008_REPLAY"
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.analysis.goal008.audit \
  --output-directory "$GOAL008_REPLAY/audit"
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.analysis.goal008.pipeline \
  --audit-gate "$GOAL008_REPLAY/audit/source_gate.json" \
  --destination "$GOAL008_REPLAY/phase1" \
  --code-commit 6a5b2b99f4566bbd7dde5dbc7c44c866b19fe802
.venv-server-v2-goal007-plots-20261016/bin/python -B -m offline_evaluation.analysis.goal008.examples_cli \
  --phase0-gate "$GOAL008_REPLAY/audit/source_gate.json" \
  --phase1-manifest "$GOAL008_REPLAY/phase1/PHASE1_MANIFEST.json" \
  --destination "$GOAL008_REPLAY/examples" \
  --code-commit 7bd5183cdcab7506c25d163f74d45075cdec700c
.venv-server-v2-goal007-plots-20261016/bin/python -B -m offline_evaluation.analysis.goal008.publication \
  --goal007-directory reports/server_only/goal007 \
  --phase1-directory "$GOAL008_REPLAY/phase1" \
  --audit-directory "$GOAL008_REPLAY/audit" \
  --output-directory "$GOAL008_REPLAY/publication" \
  --examples-manifest "$GOAL008_REPLAY/examples/figures/example_manifest.json"
```

今回の出版処理はさらに読取り専用原本lock、Python通信遮断、前後の全原本hash検証で囲んで実行した。[実行証跡](publication_execution_verification.json)を参照。GitHubへのコード保存通信は、研究API送信と区別している。
