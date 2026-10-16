# Goal010 v2 — 完了済みGoal010への指標監査追加仕様

POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC。これは既存結果を見た後の **post-hoc pre-computation specification** であり、事前登録ではない。旧Goal010はHEAD `080df2be3c5e949ff9bab6029d8b349fa7ed84d0`で完了しているため、v2改訂冒頭に従い、別ディレクトリ `goal010v2`、別run `goal010-v2-<UTC>` に新指標の監査だけを追加する。既存のPhase0–5は再定義せず、hashを再検証した原本・報告に対応付ける。新API生成、モデル照会、token-count API、キー検査、人手実験、依存導入、OS/SSH/ユーザー変更は全て0。既存DraftはGET照合だけとし更新しない。

## 固定対象と計算順序

原本はGoal007の固定960 slots（934 valid、25確定出力失敗、1 HTTP520不明）を全て残す。940の承認retry選択、元401と942のunknown-usage保留、961固定attempts、991歴史callsを変えない。独立unitは固定60建築。4 canonical sourceはverified manifest/plan/ledgerから解決する。既存Goal008の180scores/360探索とGoal007の20確認比較、全旧code/hashは凍結する。

旧Goal010の67 report products、20 code files、6 phase receipts、480 baseline actions/arraysを毎段階再検証する。旧MANIFEST SHA256は `dcad6f41f991425063c239798fec9b8d3bec2c0474fde5f901205a67d1432d7f`、元source package SHA256は `01fcabeac3c37fd39868d90234781773ddf6215cca1f1af5fb331111139f9b51`。旧runのパスは公開コピーreceiptから取得し、手入力した別原本へ切り替えない。

1. stage0: shared read-only14 locks、実行中生成プロセス監査、Python network接続禁止、原本/hash/committed code/spec再照合。source gateを保存する。
2. stage1: `buildings_100_v4_building_034` の固定16 slotsとO/Pの8 baseline（filled/shell）をpilotする。形状を結果で選ばない。
3. stage2: 034と残りscene_id辞書順の先頭9建築、計10建築・160 slots・80 baseline。stage1の完了receiptと全出力hashを確認してから実行する。
4. stage3: 全60建築の960 slotsと480 baseline。既存成功voxelsをread-onlyで読み、保存parsed actionsとの一致・GT YXZ→XYZ world・stored strict shift certificate・旧exact metric全欄を独立実装で検算する。旧validated metric objectsには書き込まない。
5. stage4: 全新metricの分母・対応比較・順位変化・reliabilityを固定統計規約で作成する。source/code/prior productsを再照合してreceiptを閉じる。

各段階は前段階のPASS receiptを要求する。既存出力先・partial stageの上書きを拒否する。source/コード/数値不一致はSTOPし、以前のstageを保持する。metadataやlockファイルだけでrunningと判断せず、実際のprocess handleを確認する。

## 指標と欠測

[METRIC_SPEC](METRIC_SPEC.md)と[METRIC_REGISTRY](METRIC_REGISTRY.json)に式・分母・値域・座標・shift・null・限界・実装hashを定義し、解析前にcommitする。新しい合成scoreは作らない。最優先coreはoccupancy precision/recall、surface F1@1/2、conditional exact material accuracy、XZ footprint IoU。補強としてvolume/column/top_y/multiscale、6-neighbor connectivity、alignment displacementを記録する。@8は拡張診断、26-neighborは今回実施しない。6-neighborは部屋/通行可能性ではない。

全960行をJSON/CSVに残し、元statusと新metric_statusを分離する。truncation/validation/operational missingの幾何metricはNULL+reason。既知の確定モデル出力失敗だけ、registryで明示したhigher-better[0,1] metricの補助failure0表に0を付ける。正常なempty output、C=0のconditional material、resource limit、GT欠落、401/520不明はこの置換に含めない。無限定誤差やrepairは0補完しない。

O/Pのbaselineは旧Goal010の凍結された同じ480予測を利用し、新規構造生成・GTによる調整は行わない。O構築はdimensions_estimateのみ、Pは開示world_bboxのみという旧input boundaryとfixtureを継承検証する。新geometry metricsに同じ規約を適用する。人工stoneの材質metricはNULL `ARTIFICIAL_BASELINE_MATERIAL_NOT_INTERPRETED` とし、Builderとの材質能力比較から外す。

## 独立検算とCPU上限

新engineの表面KD-treeに対し、独立実装で6近傍数からboundaryを作り、dense Euclidean distance transform（bbox unionが8,000,000 cells以下）または距離1/2の有限7/33整数offset照合を使う。整数集合/count/投影/floor poolingも別実装で検算する。旧exact実装とのg,p,C,M、IoU/F1、repair、dimensionの整合を934成功全件で要求し、baselineの旧absolute/aligned IoU480件も一致を要求する。k=1はstrict IoUに一致、全追加metricで保存τ固定。shiftXYZ/L1/L2とoccupied bbox比を記録する。

既存venv、single-thread BLAS/OpenMP/MKLを用いる。engineの固定stage budgetと512MiB作業量推定、occupied/surfaceの各2M上限はMETRIC_SPECに従う。pipelineは1sample CPU180秒/wall240秒、1解析stage CPU3600秒/wall7200秒、address-space4GiB、開始時空き10GiB以上を固定する。超過時は近似・縮小・0補完をせずNULL reasonを残す。stage予算超過はSTOPであり完了receiptを作らない。計測対象コマンドCPU秒/elapsed/RSSを報告し、対話時間と区別する。

synthetic反証fixtureを先に実行する：同一/shift/全材質不一致、1block表面距離境界、filled/shell、幅1/2、負座標floor、empty、連結成分、重複・軸・P hiddenGT拒否、failure0不可条件、建築cluster、固定shift・same GT結合、path/hash/network blockade。specとproduction実装hashをcommitした後にのみ実データpilotを実行する。pilotデータで閾値や形状を最適化しない。

## 統計・旧phaseの再利用・保存

[STATS_SPEC](STATS_SPEC.md)に対応付けとbootstrapを固定する。原則Builder−baseline、building内の適格説明差を平均後、60 planned buildingsを等重み、PCG64 seed20261017、2000 draws、type7 percentile95%探索CI。両Builderや説明providerを独立Nとして加算しない。common-valid N/除外理由を全セル保存。順位変化は同一の適格入力集合に揃えて比較し、異なる欠測集合の見かけの変化を区別する。新p/Holm familyは作らない。

旧Goal010の180 request/6views監査、36provider対比、60CPU exact control/101E診断、可視性NOT_VALIDATED、proxy1350結果は、改訂の指標追加以外の要件の根拠としてhash付きで参照する。新EのM/C・M/g・macro recall・全条件reliabilityを追加する。撮影のactual yaw/pitch・投影/遮蔽の独立キャリブレーションが不十分なため、rendered silhouette/visible-face metricsを数値化しない。boundaryは可視外表面ではない。

E容量診断は元の120件の保存済みcapacity CSVをそのまま用い、Builder別に、GT occupied数・参照cuboid命令数・exact spec UTF-8 bytesの3横軸と観測output tokens縦軸を固定したSVG散布図で示す。valid/truncationを分け、plan上限32768を併記する。回帰・token換算・未完了出力の完成予測は行わない。成功率、valid-only、definitive failure0は別表の別の問いとして保持する。

新runには全サンプルのprovenance・source SHA・prediction SHA・実装/仕様hash・independent checksを保存する。Gitは新code、小さなaggregateと報告のみ、raw response/GT/予測voxelsを追加しない。旧成果物は上書きしない。最終報告にmetric比較で結論が変わる範囲と、変わらない因果的限界を記す。保存は既存専用branchだけ、main/reset/rebase/forceは行わない。
