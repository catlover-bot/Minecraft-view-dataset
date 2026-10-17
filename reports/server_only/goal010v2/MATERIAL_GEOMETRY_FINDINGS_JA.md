**Goal010v2：材質配置と幾何の診断所見（全数照合済み）**

占有形状が完全に一致しても、厳密材質配置が一致するとは限らない。Eの成功101件中、占有IoU=1は50件、そのうち26件に材質不一致があった。一方、正常な空予測1件ではPrecisionや条件付き材質正解率が定義できず、成功行として残したままNULLを保持する。以下は固定された全960 slots・既存480 baselineの保存値と集計を照合したCPU診断である。

解析ラベルは`POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC`。凍結commitは`d862a55134bba85a6a5e9677bcc6e94c719ca1aa`。[計算前仕様](METRIC_SPEC.md)と[registry](METRIC_REGISTRY.json)に従い、全新指標は同じ保存済みstrict occupancy shiftを使う。新しい位置合わせや生成はなく、Goal007の20比較、Goal008のproxy・360解析、旧Goal010の成果物を置換しない。

**全960行の状態と独立な建物数**

| 条件 | Builder | 計画slots | 成功/計測済み | 成功の棟数 | validation失敗 | truncation | 運用欠測 | 正常empty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O | OpenAI | 180 | 177 | 60 | 3 | 0 | 0 | 0 |
| O | Anthropic | 180 | 179 | 60 | 1 | 0 | 0 | 0 |
| J | OpenAI | 180 | 177 | 60 | 2 | 0 | 1 | 0 |
| J | Anthropic | 180 | 180 | 60 | 0 | 0 | 0 | 1 |
| P | OpenAI | 60 | 60 | 60 | 0 | 0 | 0 | 0 |
| P | Anthropic | 60 | 60 | 60 | 0 | 0 | 0 | 0 |
| E | OpenAI | 60 | 55 | 55 | 0 | 5 | 0 | 0 |
| E | Anthropic | 60 | 46 | 46 | 0 | 14 | 0 | 0 |

合計934成功を計測し、25モデル出力失敗（validation 6・truncation 19）とoperational missing 1を含む26行はNOT_EVALUABLEとして保持した。追加のresource limit・optional未計測・not-runは0。独立unitの計画数は60棟。O/Jの3説明providerや2 Builderを独立N=180/360へ水増ししない。Eの単群成功平均は55棟と46棟で集合が異なり、両Builder共通成功46棟の対応比較とは区別する。

baselineはO180入力×2形状=360、P60入力×2形状=120、計480の既存予測を計測し、全件で元のIoUと一致した。新規形状生成やsceneごとの最適化はない。人工材質による5材質scalarは全480件×2frameで適用外NULL、材質理解の比較対象にはしていない。

**成功行のNULLは失敗と分ける**

| scalar（成功934件中） | absolute有効数 | aligned有効数 |
| --- | --- | --- |
| conditional_exact_material_accuracy | 927 | 933 |
| common_top_y_mae | 929 | 933 |
| occupancy_precision | 933 | 933 |
| surface_precision_at_1 | 933 | 933 |
| surface_precision_at_2 | 933 | 933 |
| common_footprint_pred_coverage | 933 | 933 |
| pred_largest_component_fraction6 | 933 | 933 |
| その他38 scalar（M/g・macro・材質IoU・占有Recall/F1含む） | 934 | 934 |

slot715＝building_001 / OpenAI説明 / Anthropic Builder / Jは、正常に完了した空予測である。占有Recall/F1/IoU、GT材質recall、surface F1は0だが、予測数を分母とするPrecision、C=0のM/C、共通footprintのないtopYなど7値はNULL。これは25モデル出力失敗へ追加しない。absoluteではさらに位置が重ならない非空成功6件があり、M/CのNULLは計7件。共通footprintなしは計5件で、C=0と同義ではない。

942はJ/OpenAIのHTTP520運用欠測として保持する。補完940の選択履歴と過去401/520のunknown usage予約は変更しない。failure0補助表は既知の完了済みモデル出力失敗かつ固定whitelistに限る。条件付きM/C、正常成功のNULL、942/401のunknown、resource・GT・baseline欠落へ0を代入しない。全体でfailure0適格25件を加えても、Precision等は正常emptyを残すため958値、その他の対象bounded指標は959値となり、960へ埋め切らない。個別の条件・provider・metric分母は[2640行の分母表](analysis/4_summary/metric_denominators.csv)に保持される。

**占有、条件付き材質、GT全体の材質recallを分ける**

g=GT占有数、p=予測占有数、C=共通占有数、M=共通位置で厳密材質IDも一致した数。占有Recall=C/g、条件付き材質正解率=M/C、GT材質recall=M/g、材質IoU=M/(g+p−M)。C>0の各sampleでM/g=(M/C)×(C/g)だが、群平均を掛け合わせることはできない。GT材質macro recallは各GT IDのrecallをID等重みで平均し、体素頻度で重み付けするM/gとは別の問いを測る。

次表は保存済み`metric_denominators`のaligned・common_valid単群平均をそのまま掲載する。O/Jは建物内の利用可能な記述を平均してから建物を等重みで平均する。これは両Builder間の対応差やCIではない。M/Cのn以外の材質値は成功数を分母とする。

| 条件 | Builder | M/Cのn/N | 占有Recall | M/C | M/g | GT材質macro | 材質IoU |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O | OpenAI | 177/60 | 0.274386 | 0.130732 | 0.037871 | 0.025333 | 0.021005 |
| O | Anthropic | 179/60 | 0.334572 | 0.111265 | 0.036491 | 0.025902 | 0.016857 |
| J | OpenAI | 177/60 | 0.257525 | 0.133969 | 0.036956 | 0.023118 | 0.021018 |
| J | Anthropic | 179/60 | 0.327413 | 0.115546 | 0.039781 | 0.026345 | 0.018031 |
| P | OpenAI | 60/60 | 0.298269 | 0.362172 | 0.108840 | 0.058189 | 0.056267 |
| P | Anthropic | 60/60 | 0.867373 | 0.162772 | 0.140656 | 0.112061 | 0.024269 |
| E | OpenAI | 55/55 | 0.923055 | 0.933593 | 0.861242 | 0.846187 | 0.827372 |
| E | Anthropic | 46/46 | 0.992453 | 0.867522 | 0.861291 | 0.872640 | 0.807393 |

O/Jでは重なる位置だけのM/Cより、GT全体を分母にするM/gが小さい。P OpenAIもM/C=0.362172に対しM/g=0.108840であり、重なりの選択を含む条件付き値だけでは再建築全体の正確さを示せない。材料をどこかに含むmaterial-set F1とも同一視しない。値の違いは語彙・配置・占有の複合であり、ここではalias補正や因果分解をしていない。

**E全数とbuilding_034の厳密な材質反例**

| E Builder | 成功 / 計画60 | 占有IoU=1 | 占有1かつ材質不一致 | 占有・材質とも完全一致 |
| --- | --- | --- | --- | --- |
| OpenAI | 55 | 36 | 22 | 14 |
| Anthropic | 46 | 14 | 4 | 10 |

失敗したE 19件の「完成した場合の品質」は推定しない。成功率とvalid-only得点を黙って掛け合わせず、成功55/60と46/60を別に示す。

building_034では両E結果ともτ=(0,0,0)、g=p=C=4912。absolute/alignedの占有IoU、footprint IoU、surface F1@1/@2はすべて1。

| EのBuilder | M | M/C | M/g | 材質IoU | GT材質macro |
| --- | --- | --- | --- | --- | --- |
| OpenAI | 1794 | 0.3652280130 | 0.3652280130 | 0.2234122042 | 0.6666666667 |
| Anthropic | 4912 | 1 | 1 | 1 | 1 |

OpenAIの不一致3118個はGT sandstone→予測glassが2958個、GT glass→予測brick_blockが160個。brick_block568、quartz_block1033、sea_lantern3、stone_slab190は一致した。M/CとM/gの一致はこの例でC=gだからであり、材質IoUの分母は4912+4912−1794=8030になる。同名材質が予測の別位置にあっても一致には数えない。

macroでは6 GT材質中4材質のrecallが1なので4/6=0.666667。3個のsea_lanternも2958個のsandstoneも1 ID分の重みを持つ。一方、体素頻度を反映するM/gは0.365228。この差を新しい良否順位やどちらか一方が「真の正解率」という主張へ変換しない。[全材質混同行列](analysis/3_full/exact_material_confusion.csv)と[材質別頻度・recall](analysis/3_full/gt_material_frequency_recall.csv)を保持した。

建物間でも重みは異なる。E成功行のcountsをプールしたΣM/ΣgはOpenAI225135/271399=0.829535、Anthropic187246/215591=0.868524で、上表の棟等重みM/g平均0.861242/0.861291とは違う。プールしたΣM/ΣCはそれぞれ225135/241132=0.933659、187246/214140=0.874409。これらは大きいGTや共通占有の多い建物を強く重み付けする参考算術であり、主集計の重みを差し替えない。

**保存shiftの効果と幾何対照**

| 条件 | Builder | 成功n | 非zero shift | L1中央値 | L1最大 | IoU増加中央値 | IoU増加最大 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O | OpenAI | 177 | 177 | 15 | 65 | 0.087489 | 0.266726 |
| O | Anthropic | 179 | 178 | 16 | 51 | 0.085598 | 0.314154 |
| J | OpenAI | 177 | 177 | 15 | 50 | 0.091215 | 0.299437 |
| J | Anthropic | 180 | 179 | 16 | 54 | 0.083749 | 0.265959 |
| P | OpenAI | 60 | 60 | 7.5 | 22 | 0.079233 | 0.228933 |
| P | Anthropic | 60 | 39 | 2 | 15 | 0.003054 | 0.110647 |
| E | OpenAI | 55 | 0 | 0 | 0 | 0.000000 | 0.000000 |
| E | Anthropic | 46 | 0 | 0 | 0 | 0.000000 | 0.000000 |

これは保存済み各行の記述分布で、反復行を独立標本とした検定ではない。全934成功でIoU増加は非負、最大L1は65 blocks。E成功101件はすべてshift0。O/Jでは多くの行がGT参照の平行移動で改善する。aligned値は生成モデルが世界原点を特定できた証拠ではなく、変位の大きさも画像だけで原点を識別できるかに依存する。[XYZ/L1/L2とoccupied bbox比](analysis/3_full/alignment_displacement.csv)を残した。

building_034のP Anthropicの保存形状指標はP filled-bbox baselineと一致する。Recall=1でもPrecision=IoU=0.1218253968、volume ratio=8.2084690554であり、GTを覆うことと余剰占有を抑えることは別である。同じ1棟のO/OpenAI説明/OpenAI BuilderはIoU0.198611でfilled0.199349をわずかに下回るが、surface F1@1は0.467850でfilled0.437238を上回る。これを全体へ一般化せず、[全paired効果とCI](analysis/4_summary/metric_baselines_paired.csv)および[共通対を使った順位差診断](analysis/4_summary/metric_rank_reversals.csv)を参照する。

**表面・粗視化・列・連結性の限界**

surfaceは空洞の内壁も含む6-neighbor boundaryのセル中心距離であり、画像から見える外表面ではない。building_034のGT4912個すべてがboundaryでも全て可視とは言えない。GTの連結成分は全60棟で26～62、予測の成分は成功934行で0～76だったが、部屋数・通行可能性・穴埋めの正しさは測らない。26-neighbor感度は仕様どおり任意項目として省略した。

topYは共通投影、column count MAEは投影unionを使う。欠けた列を隠す共通部分だけの低い誤差を避けるため、common coverageとGT/予測/union列数を併記する。すべての非NULL値は独立な列走査・表面EDT/有限offset・coarse set照合に合格した。

下表は同じ934成功に対する二つの固定anchorの絶対差で、保存値の分布要約である。k8は拡張診断であり、高いcoarse IoUを主要な成功判定にしない。k1は全件で同frameの元IoUと一致した。

| k | frame | anchorで値が変わる / 934 | 絶対差中央値 | 絶対差最大 |
| --- | --- | --- | --- | --- |
| 2 | absolute | 867 | 0.012910 | 0.221178 |
| 2 | aligned | 872 | 0.015727 | 0.124918 |
| 4 | absolute | 857 | 0.024402 | 0.287533 |
| 4 | aligned | 859 | 0.034967 | 0.215091 |
| 8 | absolute | 844 | 0.066304 | 0.419592 |
| 8 | aligned | 846 | 0.064424 | 0.419924 |

粗視化は誤った隙間や細部を消し、anchor感度はk8で最大約0.42に達する。表面許容距離・footprint投影・coarse occupancyはそれぞれ別の誤差に鈍感なので、元占有P/R/IoUと併記する。全体のbaseline比較での勝敗を一つの指標だけで言い切らない。可視面・silhouetteの校正は成立していないため、その数値や「不可視部分が主原因」という結論は追加しない。

**検証範囲と原本の保全**

canonical run: `/root/minecraft-runs/goal010-v2-20261016T233143Z`。stage3 receipt PASSが束縛した36 productsのbyte数・SHAを再照合した。934成功と480 baselineを全件計測し、元指標、保存τ、world YXZ/XYZ、独立幾何照合はPASS、optional未計測0。stage4 PASSの分母2640行と信頼性8行もreceiptのSHAに照合し、本稿の単群平均と一致を確認した。新指標計測器・bootstrapを本稿作成時に再実行していない。

stage3記録はCPU1268.4265秒、wall1268.7048秒、最大RSS413024 KiB。これは当該全数計測stageの実績であり、OS隔離の保証や全工程の累積時間とは異なる。研究API・モデル・token-count・キー照会・人手実験・checkpoint変更・Draft更新は0。原本source packageは前後一致。旧RQ2はNOT_TESTABLE_WITH_CURRENT_MATCHED_SCORESのままで、新しい材質・幾何診断をcaption品質の妥当性証明へ読み替えない。

全960行JSON SHA-256: `1d15032a58f9eacb272b87eb2a425fc283266d9ec596e283aa2dddb3d97ae54e`。材質・幾何詳細JSON SHA-256: `3ee74d3fcb45032cbabe834fc33b23285cc3aa9cd043cd99abc8eb7ae5a7c3db`。共有用の大きいJSONがgzipでもcanonical JSONは保全し、[全960行CSV](analysis/3_full/new_metric_values_960.csv)から同じslotとframeを追跡できる。
