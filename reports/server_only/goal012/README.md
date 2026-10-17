**Goal012 — 新規性・元のRQ・反証可能性の研究監査**

**現在の証拠は限定的な診断ケーススタディを支えるが、元のRQ2/RQ3を実証していない。** Goal011の「A：追加有料実験なし」は、この限定的な報告方向A0に対して維持する。元のARR向け中心仮説まで達成したという意味には拡張しない。論文・Abstract・LaTeX・投稿資料は作成していない。

最も強い観測は、Exact-GT入力でも保存Builderに下流失敗が残ることである。E valid101（OpenAI55／Anthropic46）の完全geometryは36/55・14/46、その中に材質不一致22/36・4/14がある。19 truncationは別扱い。034/OpenAIではC4912・M1794、GT-informed対応上界M4752・残差160。画像説明の欠落だけで全失敗を説明することはできないが、各原因の寄与や内部機構を識別したわけではない。

O/allの同一情報filled-box対照は各Builder独立N60、適格ペア177／179。Builder−boxのaligned IoUは−0.011960／−0.011615、precisionとsurface F1@1は正、recallは負。Gemini説明のIoU差は正、absolute precision差は負という反例も維持する。これは評価対象・情報源・frameに依存する観測で、世界配置、可視外観、機能や一般的空間能力の優劣を意味しない。[全判定と既存出典](RQ_EVIDENCE_GATES.md)。

| 元の問い | 現在の判定 | 不足する証拠 |
| --- | --- | --- |
| RQ1：保存画像説明からの再構築 | PARTIALLY_TESTED | 固定モデル設定・60建築の幾何／材質／信頼性は直接測定。一般化、外観／機能は未検証。 |
| RQ2：従来の独立品質による再建築予測 | NOT_TESTABLE | 同一原文に厳密対応した従来品質score。Goal008のGT属性proxyは代替不能。 |
| RQ3：同一情報prose対Structured IR | NOT_TESTABLE | Oは既にJSON、Jはlossless再シリアライズ。単発生成でdecodingも明示制御されていない。 |
| RQ4：image→text固有損失の因果分離 | NOT_TESTABLE | 情報内容、可視性、Builder、出力容量・転記を分けた制御介入。P/EのGT追加差では代替できない。 |

指定4論文と追加6論文の一次本文を確認した。text→3D、MinecraftのJSON IR、再構築型caption評価、geometryと材質の区別、P/R/surfaceによる評価の違いは既出である。特に [T2BM](https://arxiv.org/abs/2406.08751)、[RICO](https://aclanthology.org/2025.emnlp-main.1105/)、[CTTD](https://arxiv.org/abs/2607.23235)、[Tatarchenko et al.](https://arxiv.org/html/1905.03678v1) との競合が大きい。残る候補は、既知block GT・同一入力対照・全valid Eと打切りを区別した原本追跡可能な診断の組合せであり、新規性の十分性や世界初は未確定。[比較行列](PRIOR_ART_COLLISION_MATRIX.md)、[貢献候補の順位・査読反論・反証条件](NOVELTY_AND_FALSIFIABILITY_JA.md)。

**推奨はA0：exact-GTでも残る下流失敗を中心とする限定的診断にとどめ、有料生成を停止すること。** A1として元RQ2を中心にするなら独立品質の原文対応が必要だが、新Builder生成が必須とは限らない。A2としてprose/IRやimage→text因果を中心にするなら新たな制御介入が必要になる。費用・N・反復数は未確定で、現在の予算・旧991 callsは新承認ではない。[条件付き実験判断](EXPERIMENT_DECISION_JA.md)には概念行列だけを置き、実行器・paid plan・approval等を作成していない。rest30反復だけでは元RQの欠損を解決しない。

**保全・検証。** 開始HEADは指定どおり `86a154b67a53d00de61f07fa006832204f7dec8a`、開始worktreeはclean。既存18,239 tracked files、Goal011の保護inventory・rest inventory・canonical成果物を合わせた28,286ファイルを前後照合した。Goal007の20比較、Goal008の360探索、予算・plan/ledger/authority・評価原本を変更していない。960 slots／934 valid／25失敗／HTTP520欠測1、961 attempts、歴史的991 calls、unknown-usage予約2件を維持した。新しい数値実験、p値、bootstrap、指標、統合スコアは0。既存602 testsを本Goalで再実行したとは数えず、hash・原文対応・リンク・記述の独立レビューを検証に用いた。[INTEGRITY.json](INTEGRITY.json)。

変更範囲は `reports/server_only/goal012/` の7報告書のみ。外部本文のアクセス範囲、未確認事項、原本の完全hash、参照commitは [SOURCES.json](SOURCES.json)。公開資料の閲覧とGitHubへの許可済み保存を、研究生成APIと区別する。**research_api_requests_sent=0、人手実験0、モデル／token-count／キー・認証情報照会0**。次のGoalの作成・実行は行わない。
