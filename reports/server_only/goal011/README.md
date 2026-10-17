# Goal011 — 因果主張・材質記号対応・rest30事前監査

**全5段階完了。結論は (A) 追加有料実験なし。** 現在のデータで守れる主張を、保存された再建築の失敗と指標依存の順序に限定する。解析はすべて `POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC`。将来案は `UNAPPROVED_NOT_RUN_NO_LAUNCHER` で、事前登録や課金承認ではない。

研究API・モデル照会・token-count API・キー照会・人手実験・新生成・retry・checkpoint/Draft変更は0。元の60建築／960 slots／934 valid／25出力失敗／HTTP520欠測1を維持した。歴史的991 calls、fixed-scope961 attempts、unknown-usage hold2件、global940の承認済みretry成功、正常empty成功1件も再分類していない。

**baseline監査。** Oは説明のdimensions_estimateと固定原点、Pは供給world_bboxだけから構築。GTはBuilderと対照のstrict aligned評価に同じ規則で使用され、実world位置の正しさを測ってはいない。元の入力・GT・shift・有効mask・建築等重み集計は一致し、重大な数値矛盾は0。以前からの2件の約10⁻¹⁷の丸め差を保存し、旧表は変更しなかった。

O/all／filled-box／alignedの差（Builder−対照）は以下のとおり。各Builderの独立Nは60建築、適格ペアは177／179。95% CIは既存Goal010v2の探索的保存値で、新CI・p値は計算していない。

| 指標 | OpenAI：差 [保存95% CI] | Anthropic：差 [保存95% CI] |
| --- | --- | --- |
| occupancy IoU | −0.011960 [−0.020352, −0.003229] | −0.011615 [−0.019086, −0.004501] |
| occupancy precision | ＋0.092574 [＋0.069295, ＋0.111379] | ＋0.065285 [＋0.049145, ＋0.080604] |
| occupancy recall | −0.211233 [−0.234255, −0.185203] | −0.153559 [−0.176890, −0.129602] |
| surface F1@1 | ＋0.099288 [＋0.083128, ＋0.116816] | ＋0.064520 [＋0.048611, ＋0.081275] |

箱の被覆とBuilderの選択性の違いに整合するが、全指標・全条件での優越性ではない。全5サイズ層のO/allでは符号を維持する一方、Gemini説明のIoUはBuilder側が正、absolute Precisionは両Builderで負だった。core4指標・感度6指標のmask差異は160群中0。footprintの全体CIは0を含む。surfaceは空洞内壁を含む6近傍boundary voxel中心の距離で、可視画像面ではない。

**E材質診断。** 101 valid（OpenAI55／Anthropic46）を全件検算し、未計算0。19 truncation（5／14）は別枠。完全占有geometryは36/55・14/46、そのうち材質不一致が22/36・4/14。CPU control60/60はモデル成功に加算しない。

034/OpenAIは `g=p=C=4912, M=1794`、不一致3118、元material IoU `0.2234122042`。系統的置換に加えてラベル合流があり、GT-informed一対一／many-to-one上界はともにM4752、残差160だった。元予測は修正していない。平均M/Cは元→一対一上界→many-to-one上界の順にOpenAI `0.933593→0.974694→0.983818`、Anthropic `0.867522→0.970875→0.974070`。これらは各成功集合の診断上限で、モデル性能や内部機構の証明ではない。

**rest30。** 30 unique・各層6・eval60/smoke10と重複0。3 provider分の説明90件、保存候補画像360件、90保存requestの各6画像をhashへ束縛した。確認済みGoal005–010の固定planでは未評価だが、旧alignment解析・v5結果には全30の同ID参照がある。1GiB探索上限による未検証ファイルも残り、完全未見や過去未評価とは認定できない。

将来を再検討する場合の第一候補R1は30×2 Builders×OpenAI説明×r1＝60 calls、予約上限19,966,080 tokens。R3は180 callsで、反復分散が未同定のため追加費用を正当化する精度利得は未確認。主estimandはaligned IoU差、surface F1@1はcritical counter-metric。N30の十分性は保証せず、費用は `REQUIRES_CURRENT_PRICE_VERIFICATION` のまま。実行器・authority・ledger・approvalは作成していない。

因果的なimage→text情報損失、従来caption品質RQ2、visibility/occlusion、美的・機能的建築忠実度は現在のデータではUNTESTABLEまたはINDETERMINATE。O自体がJSONでJはlossless reserializationであり、同じ説明からの反復生成では因果問題は解消しない。[最終decision gate](RESEARCH_DECISION_JA.md)は6種類の主張を分け、最も強い反対解釈と再検討条件も記載した。

**来歴と保全。** 指定の旧HEAD `d973269219f73a3ec52f94fe4508385681696114` に対し、開始時は質問・計画だけをcommitした `f91e62331df57a42404f6a9937669d7461586002` と未追跡コードが存在した。その差を確認して引き継いだ。新規実データ計算より前にコードを別commitし、全stage receiptへ結び付けた。

| binding | 値 |
| --- | --- |
| branch | `research/server-only-20261013` |
| 質問・計画commit | `f91e62331df57a42404f6a9937669d7461586002` |
| 実行code commit | `93eaa3f8a06d0e32415fc4f565bb06b75ddcac70` |
| ANALYSIS_PLAN SHA256 | `f2d630c836c5fa2a7ae7aff4dc886866791c1a0e37893c68a7f92cf260e2b963` |
| QUESTION_REGISTER SHA256 | `e110190d7d3b3fb130bc35eaf9b1aa225c65c3c192fa2bd1b3563983181e422a` |
| Goal011 run root | `/root/minecraft-runs/goal011-20261017T020844Z` |
| Goal010 canonical root | `/root/minecraft-runs/goal010-20261016T223832Z` |
| Goal010v2 canonical root | `/root/minecraft-runs/goal010-v2-20261016T233143Z` |
| 継承source package SHA256 | `01fcabeac3c37fd39868d90234781773ddf6215cca1f1af5fb331111139f9b51` |

保護対象23,915ファイル、元のtracked18,179ファイル、rest source12,850ファイルを最終再照合した。元ファイルbytesとmembershipは不変。全25公開コピーはcanonicalと同一hashで、各ファイル10MiB以下。raw応答・GT配列・画像・action座標配列は新規公開していない。既存3 Draft／18 assetsは前後metadata一致をGETで確認したが、今回のasset再取得・復元証明ではない。保護はread-only locks・hash検証・Python接続禁止で、OSレベル隔離の保証とは区別する。

**検証と資源。** 602 tests passed（新Goal011 248、継承Goal010 202、Goal010v2 152）、fail/error/skipは最終0。統合時の同名moduleによるcollection error3件も保存し、`--import-mode=importlib`で解消した。5段階合計CPU333.509秒／elapsed334.444秒、最大RSS407,340 KiB。最終保全検査はCPU45.490秒／elapsed45.659秒／RSS257,028 KiB。詳細は[validation_summary](validation_summary.json)と各resource receipt。1 thread、stage CPU1800秒／wall3600秒、case CPU120秒／wall180秒、address space4GiB、既存依存だけを使用した。

**成果物への入口。**

- [質問](QUESTION_REGISTER.md)、[固定計画](ANALYSIS_PLAN.json)、[実行環境](execution_environment.json)、[開始状態](ENTRY_STATE.json)
- [trade-off説明](TRADEOFF_EXPLANATION_JA.md)、[claim audit](CLAIM_AUDIT_FINDINGS.json)、[全感度の読解](SENSITIVITY_REVIEW_JA.md)
- [E所見](E_SYMBOL_BINDING_FINDINGS_JA.md)、[全8例と機序の限定](E_INTERPRETATION_JA.md)、[全101件](E_CASES_101.csv)、[pooled confusion](E_GLOBAL_CONFUSION.csv)、[上界mapping](E_MAPPING_UPPER_BOUNDS.csv)、[全case confusion・証跡](E_FAILURE_MECHANISM_AUDIT.json)
- [rest feasibility](REST30_FEASIBILITY.json)、[rest sources](REST30_SOURCES.csv)、[既往利用と未検証範囲](REST30_REVIEW_JA.md)、[未承認protocol](PROSPECTIVE_PROTOCOL_UNAPPROVED.md)
- [最終判断](RESEARCH_DECISION_JA.md)、[最終保全](FINAL_INTEGRITY.json)、[公開コピー証跡](PUBLIC_COPY_RECEIPT.json)、[MANIFEST](MANIFEST.json)

CPU解析の再現には元のprivate canonical sourcesが必要で、reportだけからの完全復元を主張しない。`offline_evaluation.analysis.goal011.pipeline` のstage0→4を上記code/spec commitで固定し、新しい日時のrun rootで順に実行する。既存runは上書き不可。`--publish`は小さい監査成果物のローカルコピー専用で、GitHub Draftや研究APIを操作しない。将来の課金実行コマンドは本Goalには存在しない。
