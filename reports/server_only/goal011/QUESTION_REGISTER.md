# Goal011：少数の問いと計算前規約

分類は POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC。Goal010/v2の60建築の結果を既に見た上で定義した追加監査であり、事前登録ではない。新たな実データ算術より前に、本書とANALYSIS_PLAN.jsonをcommitする。解析コードも実行前に別commitで固定する。旧成果物は変更せず、計算は新規runだけに保存する。

## A. Oで観測された差は被覆と選択性の違いか

同一O入力からのfilled-boxとBuilderを、strict aligned occupancy IoU、Precision、Recall、surface F1@1、および独立の失敗率で比較する。surface F1@2・footprint・absolute・shell・凍結5サイズ層は感度診断とする。新総合点やp値を作らない。保存されたCIを照合し、同じ適格ペアの集合・建築内平均→建築等重みを確認する。五分位別は少数建築の記述値にとどめる。表面は空洞内壁を含む6近傍boundary voxel中心であり、可視画素ではない。

## B. Eの材質不一致は固定形状上のlabel置換でどこまで説明できるか

既存成功101件（OpenAI55、Anthropic46）を全件対象にし、truncation19件は別枠で保存する。共通占有座標の混同行列、欠落・余剰、M/C・M/g・元材質IoUを検算する。固定geometryに対する最大重みの一対一label対応と、predicted IDからGT IDへのmany-to-one対応の上界を計算し、mappingと同点候補を全件保存する。これらはGT-informed oracle diagnostic upper boundsであって修正予測・alias補正・新成功例ではない。building034は必ず掲載する。action対応は一意の非重複inclusive bounds一致だけを分類し、他はAMBIGUOUS_NOT_CLASSIFIEDとする。出力パターンから内部認知機構を断定しない。

## C. 元rest30はどの意味で独立検証に使えるか

元100建築の固定splitから30 unique、各サイズ層6、eval60/smoke10との非重複を確認する。GT・画像・説明の出典/hash、旧v5・出力・図・解析・人手研究への登場を許可範囲内で探索する。Goal005–010で未評価であることと、過去に一度も見られていないことを区別し、未確認はNOT_VERIFIEDのままにする。候補説明元はOpenAIに先に固定し、既存provenanceと共通1800-cap設計を理由とする。下流scoreが有利な説明元を選ぶ規則ではなく、欠落時の黙った代替もしない。

将来案は最大2つ：30建築×2 Builders×1説明×r=1（60 calls）、またはr=3（180 calls）。主estimandは元primaryとの連続性を保つaligned occupancy IoUのBuilder−box差とし、surface F1@1を必須counter-metricとして残す。60建築の分散は設計入力だけに使い、N30の十分性を保証しない。半幅0.03は計画上の許容幅で、妥当性が確立した最小有用差ではない。反復内分散は現データでは識別できない。価格は将来実行日に公式確認が必要で、現在のAPI照会や架空のドル見積りを行わない。計画作成は承認・台帳・実行器の作成を意味しない。

## D. 何を主張でき、何を識別できないか

失敗の存在、指標依存の順序、Builderと説明providerの設定込み比較、出力信頼性、可視性の不確実性、情報損失の因果推論を分ける。O自体がJSONで、Jはtyped treeを変えない再シリアライズである。同じO説明からの反復生成だけでimage→textの情報損失を識別することはできない。従来caption品質RQ2とCPU GT-derived proxyも混同しない。

6種類のclaim gateにPASS/FAIL/INDETERMINATEを付け、最後に(A)追加有料実験なし、または(B)一つの限定的rest30検証について将来の新承認、のいずれかを根拠付きで提案する。反対解釈と反証に必要な証拠も示す。ここで問いを固定し、新しいpost-hoc指標を際限なく増やさない。

## 保全と停止条件

研究API、モデル/token API、キー照会、人手実験、再生成/retry、checkpoint/Draft変更は0。現在HEAD d973269219f73a3ec52f94fe4508385681696114を引き継ぎ、main・既存コード/台帳/原本を変更しない。コードと報告は新goal011ディレクトリだけに追加する。重大な数値/出典不整合なら過去結果を直さず、証明と狭い修正案を保存して停止する。資源上限は1 stage CPU1800秒/wall3600秒、1 case CPU120秒/wall180秒、address space4GiB、空き10GiB以上。未計算を成功として扱わない。詳細な固定規約・成果物はANALYSIS_PLAN.jsonを参照。
