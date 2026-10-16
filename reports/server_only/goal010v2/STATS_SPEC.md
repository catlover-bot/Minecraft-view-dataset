# Goal010v2：新指標の分母・対応比較に関する計算前仕様

ラベルは **POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC**。旧Goal010の結果を既に見た上で、新しい実データ計算の前に固定する *pre-computation specification of post-hoc analyses* であり、事前登録ではない。Goal007の20比較・3 families、Goal008の180 scores・360探索分析、旧Goal010の全出力を変更しない。API・モデル・token-count・キー照会、人手実験は0件とする。

## 1. 入力と開始gate

`statistics.summarize(metric_rows, baseline_rows, *, gate, specification_sha256, registry)` はsource-bound scalar rowsだけを受け取り、GT・voxel・画像・応答ファイルを読み込まない。入力辞書を変更しない。Phase0 PASS、明示input_kind、定義凍結、指定したMETRIC_SPEC SHA256との一致、固定された昇順のplanned_scene_idsが必要。実解析では60建築で、合成fixtureだけ小さい設計を許す。

実入力は960 selected slots（934 success、25既知出力失敗＝truncation19＋validation6、global942のoperational missing1）、baselineは旧Goal010でGT評価前に保存済みの固定480予測である。過去401の失敗attemptを別slotとして追加せず、940の既存選択結果を一度だけ使用する。1建築の設計はO/Jが3説明生成元×2 Builders、P/Eが2 Buildersの16 slots。baselineはOの3説明＋Pの1入力×2固定構造の8予測で、Builder別の予測複製を独立サンプルとして数えない。

全scalar rowsでscene、condition、description_model、元status、input_id、input_sha256、gt_grid_sha256、world XYZ/Y-up/integer座標、`server_v2.2_strict_translation` protocolを必須とする。各baselineは同じ入力の両Builderとこれらを照合する。各予測の占有最適shiftは異なってよいが、同じアルゴリズム・同じ座標規約を使用し、各予測について新指標別にshiftを選び直さない。元のmodel statusと新指標のmetric_statusを分離する。CPUで指標が計算できないことを新たなmodel failureに変更しない。

各frame (`absolute`, `aligned`) はregistryの全45 frame metric IDsを保持し、NULLにはmetric単位の非空reasonを必須とする。sample source/prediction/GT/shift/implementation hashesは元scalar rowsに残し、paired rowsにはその全scalar row SHAと主要provenance hashへの参照を付ける。displacementのXYZベクトル・L1/L2・相対変位は元metric rowsに保持するが、位置原点が観測から識別できないため、好悪の順位・failure0・paired superiorityへ変換しない。

## 2. 成功・失敗・欠測とfailure0

主表はmetricごとのcommon-valid。正常な空予測のPrecision、交差0のconditional exact material accuracy、共通footprintがないtop-y誤差等は定義どおりNULLであり、成功行のNULLを0へ変えない。resource limit、GT missing、not-run、過去401、unknown942は0へしない。baseline自身の不正・資源制限もmodel failure0として扱わない。

補助policy `known_failure0` は、registryで明示された[0,1]・higher-is-better指標だけに別表で適用する。さらにslotが元の`failed`、既知definitive failure種別（parse/validation/execution/truncation）、usage_known=true、completed_request_outcome=true、HTTP401/520ではないことを独立照合し、`failure0_eligible`がこの条件と一致しなければ停止する。これは「出力失敗に固定0を割り当てる補助estimand」で、空形状の幾何計算や成功率を既存スコアへ黙って乗算する処理ではない。

固定whitelistは21指標：occupancy precision/recall/F1/IoU、footprint IoU、GT material recall、GT材質別macro recall、material-aware IoU、surface precision/recall/F1の各@1/@2（6個）、multiscale IoUのk1 anchor0・k2/k4/k8各anchor0/half（7個）。conditional exact material accuracyは、交差への条件付けの意味を保つためfailure0から除く。volume ratio、符号付きrelative volume error、counts、column/top-y MAE、coverage、connectivity、displacementにはfailure0表を作らない。

空GTは原本で許さず、指標エンジンのfixture規約だけに限定する。値が0であることとNULLの理由は別々に保存する。

## 3. 完全な対応比較grid

固定baseline構造は`filled_box`と`six_face_shell`。Oは同じ説明入力、Pは同じ開示world_bbox入力のBuilderと比較する。新指標を見て寸法・構造・原点・baselineを選ばない。baseline人工材質の5指標（材質一致数、conditional exact accuracy、GT recall、GT macro recall、material IoU）を比較から完全に除く。

45 frame metricsのうち40 geometry/count/diagnostic指標について、2 frames、2 structures、2 Buildersを固定する。Oのdescription scopeはall＋OpenAI/Anthropic/Geminiの4、Pはallの1。したがってmetric-policy当たり40集計cellsとなる。40 geometry指標のうち18がfailure0 whitelistに入り、全固定表は40×(40＋18)=**2320 paired summary cells**。その中に無差別に新しい主要評価を作る意図はない。

最小の主要探索的geometry coreはoccupancy precision/recall、surface F1@1/@2、footprint IoUに、元のoccupancy IoUの検算参照を加えた6指標。新しい材料coreのconditional exact material accuracyはBuilderの独立分母表で示し、人工材質baselineとは比較しない。occupancy F1は入力ごとにIoUの単調変換であり、独立した証拠として数えない。counts、volume、connectivity fractions、k8等は追加の記述診断である。

差は全て **Builder−baseline**。higher metricは正、lower metric（excess/omission、column/top-y MAE）は負がBuilderに有利。volume ratioとrelative volume errorは一方向の良否が定義できないためdescriptiveとし、符号付きの差を保存しても順位を付けない。GT count等のdescriptive量も優越性には読み替えない。

per-pairは固定480予測×2 Builders×2 frames×58 geometry-policy組＝**111360候補行**。全行をcanonical runに保持し、除外行には双方のmetric値・状態・NULL理由を残す。対応先baselineが未計測なら、Builder側がfailure0適格でもペアには入らない。集計にはplanned/eligible/excluded pairs、独立建築N、0割当数、両側のNULL理由別件数を含める。

## 4. 建築単位の平均差とbootstrap

O/allは建築内で利用可能な説明生成元のpaired差を平均し、その建築平均を建築間で等重みに平均する。O/providerは1建築1pair、Pも1建築1pair。metricごとにcommon-valid Nが変わることを明示する。Builder meanとbaseline meanも同じ適格pair集合・同じ建築重みで計算する。2 BuildersをN120へ、3説明をN180へ増やさない。

bootstrapは全60 planned buildings（scene ID昇順）をclusterとして復元抽出する。NumPy PCG64 seed **20261017**、**2000 draws**。同じ建築を複数回引けば、その建築内の全適格description pairsを同じ回数含める。元データで建築内平均を作ってから建築を抽出してよい。欠測建築はplanned frameに残し、各drawでは適格な建築コピーだけで平均する。分母0のdrawは無効とし、0へ置換しない。

元の寄与建築が2未満なら点推定だけを保存しCIはNULL。2000 draws中**1900以上の有限draw**がある場合だけ、type7（NumPy linear）の2.5/97.5 percentileから95% CIを計算する。不安定なら点推定・有効/無効draw数・NULL CI・statusを残す。すべての比較に同じ固定planned-frame drawsを再利用する。新p値・significanceラベル・Holm familyは作らない。

## 5. 分母と信頼性の表

Builder×O/J/P/Eの8 reliability行を独立に作る。planned slots、valid、identified failure、failure種類、operational unknown、not-run、metric_status、正常成功でも何らかのNULL metricを含む行数を保持する。model valid割合をmetricへ乗算しない。全planned slots合計は960。既知失敗25件と未知欠測1件を別々に数える。

新指標の分母表は、2 Builders×(O/J各4 description scopes＋P/E各1 scope)=20 groups、2 frames、45 metric＋21 failure0 metric＝**2640 rows**。planned/eligible/excluded slots、planned/寄与building N、NULL理由、failure0割当数、建築等重みmeanを保存する。group meanは新たな統計検定ではない。material指標のC=0、surface/resource、正常empty等の異なるNULLを隠さない。古い指標の値・denominator表をこの表で置換しない。

## 6. 指標による順位逆転の診断

各condition/description scope/structure/Builder/frameについて、occupancy IoUと残り39 geometry指標のcommon-valid **同一入力pair集合の交差**を取る。その同じ集合で双方の差を建築等重みに平均し、固定された良否方向で符号を比較する。異なるmetric-specific Nだけで順位逆転が生じたと誤解しないための規約である。

40 contexts×39比較＝**1560記述行**。双方の指標が良否を持ち、平均差が厳密に0ではなく、Builder対baselineの向きが反転した場合だけdirectional_reversal=true。数値0のtie、NULL、descriptive指標には逆転の真偽を強制せずNULLとする。同じpair集合のindex hashとN、両方の平均差を保存する。これは新しい検定・モデル順位・因果比較ではなく、測る側面により観測上の順序が変わるという反証診断である。別metric用のalignmentや新しいweighted compositeを導入しない。

## 7. 資源・検証・出力

bootstrapの作業配列は2000×60の整数count matrixで、scene数は60を上限とする。新依存を取得せず、既存NumPyのみを用いる。全candidate pairsはaggregate表より大きくなるのでcanonical runで保持し、Git保存は個別10MiB以下のaggregateと仕様・codeに限定する。元voxel・raw responseをstatistics出力へ複製しない。

返却keysは`paired_summary`, `paired_rows`, `exclusions`, `metric_denominators`, `reliability`, `rank_reversals`, `metadata`。source digest、registry digest、仕様SHA、seed・draw数、zero-query counterをmetadataへ付ける。

合成testsは、gate先行、完全設計・重複拒否、same input/GT/coordinate/protocol、正常空・unknown・資源NULL、failure0 whitelist、人工材質除外、建築内平均、planned frameでの欠測draw、type7 CI、同一pair上の順位逆転、異なる予測の保存shiftの許容、入力不変、実registryとの全grid整合を確認する。実データ結果を見て閾値・指標・比較を変更しない。
