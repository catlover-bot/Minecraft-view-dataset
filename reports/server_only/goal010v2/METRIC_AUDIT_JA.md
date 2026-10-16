# Goal010 v2：既存指標の実装・分母・限界の監査

POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC。旧Goal010は完了しているため、v2は新しい別runで行う指標監査の追加である。旧Goal007の20比較、Goal008の180 proxy・360探索、旧Goal010の成果物を変更しない。本書は実装の読取り監査であり、新指標の実データ計算が完了したという証明ではない。新計算は別の仕様・コードcommitと監査gateを必要とする。

旧Goal010の[完了manifest](../goal010/MANIFEST.json)は SHA-256 `dcad6f41f991425063c239798fec9b8d3bec2c0474fde5f901205a67d1432d7f`、canonical runは `/root/minecraft-runs/goal010-20261016T223832Z`。新metricは既存934 valid予測と固定480 CPU baselineを読取り専用で採点し、元の形状・材質・shiftを変更しない。960固定slotの失敗・欠測も行として維持する。再利用と新規要件の状態は[REQUIREMENT_COVERAGE.json](REQUIREMENT_COVERAGE.json)で区別する。

## 既存座標・材質・alignmentの規約

既存 `core.py` の座標は整数world XYZ、Yが上向き。local座標は明示originを一度加える。GT readerは保存NPYのYXZをworld XYZへ変換する。storage bboxの空気padding、非air集合のtight occupied bbox、撮影generation origin、world座標原点は別概念である。coreの寸法はoccupied bboxに基づく。

材質は生の文字列を無条件比較するのではなく、既存schemaの正規化後にexact比較する。許可されたlower-case IDの先頭 `minecraft:` を除き、`air` / `cave_air` / `void_air`を非占有として扱う。`stonebrick`と`brick_block`などの語彙aliasは統合しない。block stateを勝手に捨てる処理もない。新しいexact指標に追加の辞書・同義語・GT依存のlabel修正を持ち込まない。

`server_v2.2_strict_translation` は、予測へ加える整数shift τについて占有交差数を全整数XYZ平行移動の中で最大化する。g,pが固定なので、交差最大化とIoU最大化は等価である。同点は最小L1、その後XYZ辞書順。rotation・scale・clippingはない。空集合を含む場合は全shiftの交差が0で、τ=(0,0,0)を選ぶ。材質IoU・repairなど全aligned指標がこの単一τを共有する。材質、surface、footprintごとに別shiftを求めない。

厳密searchは疎な座標差の計数、整数NTT相関、または検証された上界一致で最適性を確定する。全候補を逐一列挙していない上界証明も、近似結果ではない。資源超過時は `AlignmentResourceLimit` を返し、`search_complete=false` / `resource_limit_reached=true` と理由を保持して成功値を返さない。新suiteでは保存済みの厳密certificateとshiftを検査して再利用する。`evaluate_pair` の歴史的defaultは `alignment_mode="bounded"` なので、無指定の呼出しを新suiteの厳密評価として使用してはならない。

absoluteはτ=0、alignedは保存τを用いる評価上の位置合わせであり、モデルが実world位置を特定できた証拠ではない。新出力にはτの符号付きXYZ、L1/L2、明示定義したGT occupied寸法に対する相対量とIoU増分を付ける。大きなshiftを一律に能力低下と解釈しない。

## 旧指標の式とエラーの意味

GをGT非air座標集合、Pτを移動後の予測集合、g=|G|、p=|P|、C=|G∩Pτ|とする。Mはその共通座標のうち正規化後の材質IDも一致する数。SG,SPは出現する材質ID集合である。

| 旧指標 | 実装上の式・分母 | 単位・方向・範囲 | 解釈の限界 |
| --- | --- | --- | --- |
| occupancy IoU | C/(g+p−C) | 無次元、↑、[0,1] | unionに対する交差であり「GTの何割を作れたか」ではない。過剰と漏れを分離しない |
| occupancy F1 | 2C/(g+p) | 無次元、↑、[0,1] | 各入力では2IoU/(1+IoU)。独立な追加証拠ではない |
| material-aware IoU | M/(g+p−M) | 無次元、↑、[0,1] | 座標と材質の組の集合IoU。置換は両側の不一致要素に数える |
| material-set F1 | 2|SG∩SP|/(|SG|+|SP|) | 無次元、↑、[0,1] | 位置・頻度を無視する。同じpaletteでも配置を全て誤りうる |
| dimension MAE | (|dGx−dPx|+|dGy−dPy|+|dGz−dPz|)/3 | blocks、↓、非負 | tight occupied寸法。位置・穴・材質・内部構造を測らない |
| add_count | g−C | voxel数、↓、非負 | GTに存在し予測にないセルを、予測修復時に追加する数 |
| delete_count | p−C | voxel数、↓、非負 | GTにない余分な予測セルを削除する数 |
| replace_count | C−M | voxel数、↓、非負 | 占有位置は合うが材質IDが異なるセル数 |
| 各normalized cost | 各count/max(1,g) | 無次元、↓、非負 | GT占有数で正規化。特に余剰生成により1を超えうる |
| total normalized repair | (g+p−C−M)/max(1,g) | 無次元、↓、非負、上限1なし | 1セル追加/削除/置換を各1とする仮想編集数。人のクリック数・時間ではない |

dimension MAEとmaterial-set F1は平行移動で変わらない。volumeも不変である。一方、occupancyや位置付きmaterial、add/delete/replaceは変わりうる。F1とIoUの単調変換は入力ごとの関係であり、建築等重み平均のF1を平均IoUの変換から求めてよいことにはならない。

両集合空なら旧coreはoccupancy IoU/F1、material-aware IoU、material-set F1を1、寸法MAEとrepairを0と定義する。GT非空で正常な空予測ならこれらoverlap/F1は0、add=g、delete=replace=0、repair=1となる。これは正常empty gridの結果で、truncation/validation失敗に空gridを仮定する規約ではない。今回の実GTは非空を前提に検証し、新指標のg=0はfixtureで別途明示する。

新指標ではP_occ=C/p、R_occ=C/gを分離し、FP=p−C、FN=g−Cを保存する。正常なp=0ではPrecisionをNULL、GT非空のRecall/F1を0とする。条件付き材質M/CはC=0でNULL、M/gは非空GTなら0である。Cが少ないとM/Cだけが高くなりうるため、C、g、p、M/gと元material IoUを併記する。GT材質ごとのrecallとmacro平均はGTに存在するID集合と分母を固定し、micro頻度集計と区別する。

## 旧primary・secondary・Goal008 proxyの区別

`analysis_v4.py` のprimaryは `aligned_occupancy_iou` と `aligned_dimension_mae`。固定familyはBuilder差8、J−O差4、P/E−O差8、計20。旧seed、bootstrap、sign-flip、Holm、分母、値はそのまま保全する。

secondary descriptiveはabsolute occupancy IoU、absolute/aligned material-set F1、absolute/aligned material-aware IoU、absolute/alignedのadd/delete/replace counts、absolute/aligned total normalized repair costの13項目。Goal007の16項目表はprimary2＋secondary13＋absolute dimension MAEである。coreにoccupancy F1が保存されていても、新たな独立したprimaryに再ラベルしない。

Goal008の180 proxyは説明のGT-derived attribute fidelityで、従来のcaption qualityではない。旧RQ2の状態は `NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES` のままである。

| 旧proxy | 固定実装 | 新exact metricと区別すべき点 |
| --- | --- | --- |
| strict_material_f1 | 記述のmaterial集合とGT targetの集合F1 | GT targetは占有頻度降順、3%以上を最大6 ID、該当なしなら上位6。全GT材質や座標配置のexact評価ではない |
| coarse_material_f1 | 同じ集合に既存normalize/coarse分類を適用したF1 | 歴史的alias/粗分類を含む。新しい位置付きexact材質へ流用しない |
| dimension_accuracy | 3軸の相対誤差平均eに対してmax(0,1−min(1,e)) | targetはstorage bboxのinclusive寸法。coreのoccupied寸法MAEと分母・構成概念が異なる |
| completeness | 固定5条件の充足数/5 | summary長≥30、accepted材質≥2、全寸法有効、elements非空文字列≥2、rebuild_hints同≥2。内容の真偽・文章品質を保証しない |
| composite | .30 strict + .25 coarse + .25 dimensions + .20 completeness | 旧固定重みの探索的proxy。新しい形状指標を重み付けして置き換えない |

旧proxyの寸法は保存された正の有限値をround-to-evenで整数化し最小1。欠測/不正軸は相対誤差1を与える欠測安全規約で、GT値による補完はない。materialの不正項目はdistinct INVALID tokenとして偽陽性側に残り、airは除外される。これらは旧定義の記述であり、新suiteのNULL規約と混ぜない。旧180値・360解析および旧Goal010の1350感度行を再計算・改変する必要はない。

## 失敗・欠測・補助failure0

旧fixed cohortは960 slot中934 valid、25 identified output failure、942のoperational missing1。401履歴と承認retry成功の940は同じ1 slotを二重計上しない。旧analysisはfailed/unexecutedのmetricsが非NULLなら拒否する。MAE/repairの失敗値はNULLである。

旧 `bounded_all_attempted_value` が受け入れるのはabsolute/alignedのoccupancy/material-aware IoUだけで、既知の完了したparse/validation/execution/truncation failureに限定する。Goal007層はknown usageとcompleted outcomeを追加検証する。試行IDがあるだけの401/520、NOT_RUN、budget停止、GT欠落、alignment資源超過は0にしない。

新suiteでbounded高値良好指標へfailure0を追加する場合は、旧関数の許可範囲を変更せず、新registryの明示whitelistと別表に限定する。正常empty予測のPrecision NULL、正常C=0のconditional material NULL、計算資源超過によるNULLを「モデル出力失敗」と取り違えて0補完しない。boundedであっても条件付き分母が成立しないvalid行はNULLのまま保持する。unbounded volume ratio、相対体積誤差、column/top_y誤差、shift、repair、距離、連結成分数へ0を代入しない。

各metricの分母はvalid slot数だけでは足りない。Precisionはp>0、conditional materialはC>0、top_yは共通XZ列、surfaceは境界集合の条件、resource判定などの適格条件と除外理由を保存する。集計は同一building/記述provider/Builder/inputの対応を維持し、建築内平均の後に建築等重みとする。失敗0補完表は元のvalid-only値を上書きしない。

## 新suiteで満たすべき構成概念と反証

最小コアはoccupancy P/R、boundary voxel surface F1@1/2、conditional exact material、XZ footprint IoU。これにcolumn occupancy error、共通列top_y、固定grid multi-scale、shift/体積診断を付け、単一の新総合点へ統合しない。

surfaceの境界は6近傍のいずれかが空の占有セルなので、空洞の内壁も含みうる。整数セル中心の距離≤1/2を使い、可視画素・外表面だけの測定とは呼ばない。整数距離の半径1/2なら7/33個の有限offset照合が可能で、巨大な全ペア距離行列は不要。固定閾値、empty規約、resource上限を計算前に凍結する。

footprintはXZに少なくとも1つ占有セルがある列の集合。column count誤差は両投影のunionを分母とし、不在側0で比較する。top_y誤差は共通列だけなので共通/unionやGT側被覆率も残す。平面図の意味・部屋数・階数はこれらから判定できない。multi-scaleはanchorを明示した数学的floorとOR poolingで、負座標・k=1一致・anchor感度・粗視化による穴の消失をfixtureで確認する。

6近傍連結成分と最大成分率は幾何的接続だけを測り、部屋・扉・通行可能性ではない。旧Goal010の可視性判定は `NOT_VALIDATED_NO_VISIBILITY_CONCLUSION`。カメラ/遮蔽の新しい独立検証がない限り、rendered silhouette、visible-face材質、可視率は数値化しない。新surface指標を可視性検証の代用にしない。

固定480 baselineにも同一geometry suiteと保存shiftを適用する。人工stoneのconditional/exact材質やconfusionは適用外を明示し、自然言語能力の比較へ使わない。Oは既存の寸法のみ、Pは開示bboxのみで生成済みの形状を保持する。IoU・surface・footprintでbaselineとの差の方向が変わるなら、都合のよいmetricだけを選ばず全固定比較を示す。

反証fixtureでは同一形状、平行移動、同占有で全材質置換、surface閾値境界、filled対shell、幅1/2、負座標floor、正常empty、非共通footprint、複数成分、baselineへのhidden GT非入力を検証する。同一形状の「1」はsimilarity系に適用し、MAE/FP/FN/repairなど誤差系は0、volume ratioは1、component数は元形状依存である。計算順はfixture→1建築/複数形状→10建築→934 valid＋480 baselineとし、未計算を成功扱いしない。

## 読み取った原実装のSHA-256

| ファイル | SHA-256 |
| --- | --- |
| offline_evaluation/server_v2/core.py | 3c240becec802957a52e9f23c8d494644cc59fa86498b60527e132ff1a40c268 |
| offline_evaluation/server_v2/alignment.py | d64e062fd9ad7ad73cc8a7a3e2a5355cdd0a1c17766876bffd09cb85b02ee446 |
| offline_evaluation/server_v2/analysis_v4.py | 33b8de23faaa73d884ffa9b88f040ed40eb83383e60f1e42fa3866ef14d553fc |
| offline_evaluation/analysis/goal007/statistics.py | ef834ea583ae640deec39aa47c6f4b332a178e2e0db5778beb089f57e4f78184 |
| offline_evaluation/analysis/goal008/scorer.py | 89a23f258817fa724fa971dee0fd6958734986614b1c3ffa7f6c2946861553c8 |

研究API、モデル/token-count照会、キー検査、新しい依存取得は0。原実装・原結果の変更は0。本書の式の監査と、新metricの原本適用・独立数値検算の完了証明は分けて保持する。
