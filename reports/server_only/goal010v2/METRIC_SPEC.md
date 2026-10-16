**Goal010v2：固定shiftによる追加幾何・材質指標の計算前仕様**

これは既存結果を見た後の `post-hoc pre-computation specification` である。
解析ラベルは `POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC`。Goal007の20比較、
Goal008の180 proxy・360解析、旧Goal010の定義・コード・成果物は変更しない。
新しい指標を加重合成した単一スコア、confirmatory p値、Holm familyは作らない。
この文書と機械可読 [METRIC_REGISTRY.json](METRIC_REGISTRY.json) をcommitで固定した後、
原本監査を通った別runでのみ実データへ適用する。

**入力と座標の契約**

`compute_metrics(gt_world_dict, pred_world_dict, shift_xyz)` を唯一の計測入口とする。
各gridは `{(x:int, y:int, z:int): canonical_non_air_material_id}` の辞書である。
整数にbool・floatを認めず、材質は既存reader/executorが正規化済みの厳密IDを要求する。
air/cave_air/void_air、`minecraft:` prefixの残存、無効IDを黙って除去・補正しない。
重複座標は辞書入力以前の既存validationが拒否する。新計測器はraw JSONやNPYを読まない。
GT読取りのYXZ→XYZ、storage bbox最低点からのworld変換は凍結readerの仕事である。
generation originを重ねて加算せず、occupied bboxとstorage bboxを混同しない。

GをGT非air占有集合、Pを予測非air占有集合とする。Yが上向き、XYZは整数セル中心。
absoluteではτ=(0,0,0)、alignedでは保存済み`server_v2.2_strict_translation`の
occupancy最大化τをそのままPへ加える。surface、材質、footprint、粗視化のための
新しいshift探索・rotation・scale・clippingは一切ない。τの最適性証明と選択済みattemptの
照合は呼出し側の責任であり、計測器が任意のτを最適だと偽装しない。

入力座標の絶対値上限は既存`Limits.max_coordinate_abs`と同じ100,000。
τは入力点差として各軸200,000以下、導出aligned座標は300,000以下になり得る。
これは入力上限の緩和ではなく、既存alignmentと同様に評価時の予測を切り捨てないためである。
全指標は同一G/P/τ、source/input/GT/pred SHAと結合する。

**出力と来歴**

返値の`absolute`・`aligned`は同じID集合のscalar辞書、`displacement`はshift診断。
`details[frame]`には次を保持する。

- `null_reasons[metric_id]`: NULLごとの固定された安全な理由文字列。
- `confusion`: GT ID、予測ID、countの行。片側欠落をJSON nullで表し、占有unionを尽くす。
- `material_recalls`: GT材質別のGT count、同座標同ID count、recall。
- `material_vocabulary`: 両入力の厳密ID unionをソートしたもの。別名対応や外部辞書はない。
- `column_counts`: 共通・union・GTのみ・予測のみの列数。存在しない側の占有数を0とする規約。
- `boundary_counts`: GT/予測boundary数、閾値1/2ごとの両方向hit count。
- `multiscale_counts`: k/anchor、各側coarse cell数、intersection/union。
- `flags`: empty GT/pred、両空fixture、failure0=false、approximation=false。

`provenance`はprotocol、world XYZ、保存τ、G/Pそれぞれのcanonical grid SHA、τ SHA、
それらの結合SHA、`metrics.py`実バイトSHA、計画メモリ見積を記録する。
grid SHAは座標順にソートした`[x,y,z,material]`の固定JSON行を逐次hashしたものである。
これは元ファイルSHAの代替ではない。呼出し側は元ファイル/input/source/GT descriptor SHA、
scene、description provider、Builder、plan、original attempt、selected slotをさらに束縛する。
各registry entryにはこのsample hash field名を記録する。

**占有と体積**

g=|G|、p=|P|、C=|G∩(P+τ)|、U=g+p−Cとする。

| metric_id | 定義・分母 | 解釈 |
| --- | --- | --- |
| occupancy_precision | C/p | 作ったうち正しい位置の割合 |
| occupancy_recall | C/g | GT占有を作れた割合 |
| occupancy_f1 | 2C/(g+p) | P/Rの調和平均。IoUとは入力ごとに単調変換 |
| occupancy_iou | C/U | intersection/union。単に一致ブロックの割合とは呼ばない |
| excess_count / omission_count | p−C / g−C | 余分／作り漏らしの体素数 |
| volume_ratio | p/g | 大小の比。1が同体積であって同形状とは限らない |
| relative_volume_error | (p−g)/g | 符号つき体積差。単純な高いほど良い指標ではない |

g/p/C/Uもscalarで保持する。量は平行移動で不変、C/Uはframeごとに変わる。
F1=2IoU/(1+IoU)を独立に検算し、F1とIoUを独立した複数の証拠としない。

**XZ投影と垂直列**

F(S)={(x,z):あるyで(x,y,z)∈S}。`footprint_iou`は両投影のintersection/union。
これは投影占有であり、外壁輪郭・部屋割り・用途の評価ではない。
`column_count_mae`はF(G)∪F(P+τ)上の平均|n_G−n_P|、単位blocks/column。
列が存在しない側のnは0。`common_top_y_mae`は共通投影だけの平均|topY_G−topY_P|、
単位blocks。topYを列数や階数と取り違えない。共通投影列数、union列数、両側の列数、
`common_footprint_gt_coverage=共通/GT列数`と`common_footprint_pred_coverage=共通/予測列数`
を保持し、共通列だけの高さ精度が良い場合の選択を明示する。

**厳密材質**

Mを共通占有のうち同じ厳密材質IDを置いた数とする。
`conditional_exact_material_accuracy=M/C`、`gt_material_recall=M/g`、
`material_aware_iou=M/(g+p−M)`を分離する。少ない重なりでM/Cだけが高くなり得る。
`macro_gt_material_recall`はGTに存在する材質IDごとの
「同座標同ID数/当該GT材質占有数」を材質ID等重みで平均する。
このmacroはGT体素頻度重みのM/gと異なる。予測のみの材質はmacroの分母へ追加しないが、
confusionの余分な占有として記録する。同義語・視覚類似性・GT依存aliasは追加しない。
定数人工材質のO/P baselineでは5個の`MATERIAL_METRIC_IDS`を
`ARTIFICIAL_BASELINE_MATERIAL_NOT_INTERPRETED`として別途NULLとし、材質比較へ含めない。
そのbaselineの詳細confusionも本物の材質理解の証拠へ集計しない。

**表面**

6近傍のうち少なくとも1セルが占有されていないoccupied voxelをboundaryとする。
空洞の内壁も含まれ、入力画像の見えるピクセル・可視外表面とは異なる。
セル中心間の最近傍ユークリッド距離を`scipy.spatial.cKDTree.query(k=1,eps=0,workers=1)`
で正確に求める。全ペア距離行列は作らない。閾値は固定の1と2 blocks、境界は`<=`。
予測boundary→GTのhit率をprecision、GT→予測のhit率をrecall、調和平均をF1とする。
`surface_precision_at_1/2`、`surface_recall_at_1/2`、`surface_f1_at_1/2`を
absolute/alignedで保持し、各方向のhit数と分母も残す。
nearest neighborの対応は一対一対応ではない。filledとshellで同じ表面得点でも内部占有は違う。

**粗視化と連結性**

各軸をfloor((coordinate−anchor)/k)へ写像し、OR poolingしたセル集合のIoUを測る。
負座標に整数切り捨てを使わない。k=1/2/4、拡張診断k=8。
anchorはworld(0,0,0)と(k//2,k//2,k//2)に固定し、k=1では同一anchorを重複記録しない。
IDは`multiscale_iou_k1_anchor0`、`multiscale_iou_k{2,4,8}_anchor{0,half}`。
k1は同frameのoccupancy IoUと厳密一致させる。k8はdiagnosticのみ。
粗視化は隙間や位置誤差を隠し、anchor依存で値が変わる。元IoU/P/Rを必ず併記する。

6-neighbor連結成分数と最大成分体積/全occupied数をGT・予測それぞれ記録する。
隅・辺だけの接触は連結しない。translation不変なのでframe間で同じ計算結果を再利用する。
部屋数・通行可能性・扉の開閉の意味指標や単純な品質順位にはしない。
26-neighbor感度は任意項目として今回は省略する。6近傍の固定診断に絞り、追加の
近傍定義による指標増加と計算負荷を避ける。省略はregistryにも明記する。

**位置合わせ変位**

τ XYZ、L1、L2、GT occupied inclusive bbox寸法を保持する。
相対量は各軸|τ|/occupied extent、L1/sum(extents)、L2/L2(extents)。
storage padded bboxを分母にしない。`occupancy_iou_gain`はaligned−absolute IoU。
位置合わせはGTを参照する評価処理であり、世界原点を生成モデルが推定できた証拠ではない。
τの大きさを単純な良否順位へ変換しない。

**空集合・欠測・failure0**

| 状態 | IoU/F1 | Precision | Recall | 条件付き材質 M/C |
| --- | --- | --- | --- | --- |
| GT非空、正常な空予測 | 0 | NULL | 0 | NULL |
| GT空、予測非空（fixtureのみ） | 0 | 0 | NULL | NULL |
| 両空（fixtureのみ） | 1 | NULL | NULL | NULL |

同じ規約をboundary P/R/F1へ適用する。両空footprint/multiscale/material IoUも1。
共通投影なしのtopYはNULL、両投影空のcolumn MAEはNULL。GT空のvolume/relative volume/
GT material recall/macro recall/相対変位はNULL。空occupancyのcomponent countは0、
largest fractionはNULL。空予測と不正・未完了・resource failureを同一視しない。

既知のdefinitive completed model-output failureだけ、別の補助表で
registry `failure0_allowed=true` のhigher-better [0,1]指標へ0を入れてよい。
占有P/R/F1/IoU、footprint IoU、GT/macro材質recall、材質IoU、surface P/R/F1、
各multiscale IoUを固定whitelistとする。C条件付き材質正解率はwhitelistから除外する。
正常成功のNULL、942/過去401のunknown、resource limit、GT欠落、baseline欠落には0を入れない。
MAE・体積・個数・変位、診断coverage/連結性fractionにも0補完しない。
計測器自体はfailure0を一切実行せず、適格判定は凍結履歴を検証した呼出し側が行う。

**CPU・メモリと失敗動作**

occupiedは各grid最大2,000,000、boundaryも各2,000,000点。
入力コピー前に`320*(g+p)`を計画scratch見積とし512 MiB超なら拒否する。
この係数はRSSの数学的上界やOS隔離の保証ではなく、pilotの実RSSと外部制限で補完する。
計測器はprocess CPU 120秒の段階budgetを持ち、大きいループや計算群の境界で確認する。
これは単一のC拡張呼出しを強制中断する仕組みではない。
呼出し側はsample CPU180秒/wall240秒、stage CPU3600秒/wall7200秒、address space4 GiBを
追加制限し、1棟の複数形状→10棟→全数のpilot gateと実測を保持する。

入力不正はValueError、core資源超過は`MetricResourceLimit`で全sampleをNULL理由付きとして
保持させる。surface/連結性の資源超過またはsurface依存不在は該当scalarだけNULL＋理由とし、
既に確定したcore値を近似や0へ置き換えない。未計算数を報告する。
新依存の自動取得、ネットワーク、research API、キー照会、GT/原本書換えはない。

**反証・独立検算**

合成fixturesで、同一grid、保存strict shiftでの回復、意図的に誤ったshiftを補正しないこと、
占有一致/材質全不一致、C=0条件付きNULL、材質macroと頻度recallの違い、
union列と共通topYの分母、surface距離1/2の包含境界とsqrt2、filled/shell、
1/2セル幅、内部空洞、negative floorとhalf anchor、正常empty、隅接触の複数成分を検査する。
乱数固定の小集合では、全ペア距離・全ペアManhattan隣接のdisjoint set・明示的列走査・
math.floorを独立実装として照合する。frozen strict evaluatorの選択τと同一位置で
occupancy/material IoU・F1・C/U・k1の一致を確認する。
資源・依存不在を注入し、NULL理由とfailure0未実施、実装/入力hashの束縛も検査する。
実原本の独立EDT/有限整数offset検算は呼出し側の別実装で行い、本moduleから最適化器を呼ばない。
