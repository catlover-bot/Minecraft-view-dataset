**Goal013の実施可能性監査を完了。採点は未実行で、ここで停止する。**

`POST_HOC_EXPLORATORY / GT_INDEPENDENT_SUMMARY_ALIGNMENT`。最終gateは `RQ2_GATE=BLOCKED_NO_SCORER`、理由は `BLOCKED_NO_VALIDATED_OFFLINE_SCORER`。`COMPUTATION_NOT_PERFORMED` であり、採点成功は **0/180**、image–text推論は **0/1080**。旧RQ2の `NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES` 判定を変更していない。

必須のsource bindingは180/180件でPASSした。固定60建築×3 description providersについて、原description、保存O入力、保存request、request指定6画像のbytes/SHAと順序を照合した。両Builderの保存Oテキスト360 bindingsは、180原descriptionのbytesと一致する。各建築の12候補画像からrequestが指定する6枚を区別し、360個の画像ファイル・1080画像参照を確認した。3提供元の画像順序・bytesとprompt hashも各建築で一致した。この照合はtransport実送信、画像の視認、可視性や遮蔽の独立証明ではない。

| description provider | 保存モデルID | binding | 非空summary | 空summary | 保存max_tokens | 保存生成日（UTC） |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| OpenAI | `gpt-5-mini` | 60 | 60 | 0 | 1800 | 2026-02-28 |
| Anthropic | `claude-haiku-4-5-20251001` | 60 | 60 | 0 | 1800 | 2026-03-01 |
| Gemini | `gemini-3.1-pro-preview` | 60 | 58 | 2 | 4096 | 2026-04-18 |

180件すべてのsummaryは文字列だったが、Geminiの `building_054` と `building_061` は空文字で、`MISSING_EMPTY_SUMMARY` とした。178非空summaryは「採点対象になり得る原文」であり、score-valid 178件という意味ではない。非空分の画像対上限は1068だが、実採点は0。summaryは原JSONから厳密に取得し、書換え・補完・raw本文の新規公開はしていない。原文文字列SHAとJSON値SHAは別に保持する。

**採点器の不足と尺度の限界。** 事前に機械的候補優先規則を記録してから、2既存venv・system packages・標準cache等を読み取り調査した。記録した探索範囲内に、検証可能な画像テキスト実装・真正weight・tokenizer・前処理・出典/ライセンス一式はなかった。モデル出典、active score ID、checkpoint、weight/tokenizer/実装SHAはすべて `null / NOT_SELECTED`。正式CLIPScoreは未採用の候補であり、cosine等の代用得点を作っていない。重み取得・install・既存venv変更は0。

実tokenizerがないためtoken長・未読率・truncation・summary代表性は `NOT_COMPUTED / NOT_ASSESSABLE_NO_SCORER`。文字数をtoken数の代用にしていない。source loaderの出力を原summaryと6画像に限定するfixtureはPASSしたが、実スコアのGT非依存性や尺度妥当性を実証したわけではない。保全目的のGT等のファイルhash照合と、採点入力は区別した。

**関連解析は実行していない。** 2 Builder別Spearman、both-valid平均、建築内順位一致率、failure0感度、bootstrapはすべて `NOT_COMPUTED`。outcome joinは0で、paired denominatorsとactual association Nは `null`（集合未形成）。source cohortのN=60を、相関解析が60建築で実行されたという報告に転用しない。未計算の相関を0や無相関と記載しない。

将来得点が得られても対象はsummary限定で、O JSON全体や人間評定の総合品質ではない。同じprompt・画像を確認できても、provider、保存モデルID、生成時期、1800対4096の生成上限、sampling指定の交絡は残る。現在の停止を覆すのは、必要な真正scorer一式と再現可能性・token代表性の検証であり、都合のよい相関値ではない。本Goalはその取得や新研究を承認しない。

**実行・検証・保全。**

- code/protocol事前commit：`232963b46b9cbd975f77f9f0401e5f57b0e15cc9`。
- run root：`/root/minecraft-runs/goal013-20261017T111808Z`。新規作成のみ許可し、既存runへの上書きは拒否。
- 56 fixtures PASS、fail/error/skip 0。sourceの欠測・改変・画像差替え、payload境界、規約commit照合、採点済みデータ混入、外部書込み・終了時検査失敗を確認。実モデルのreplayや平均不変性の検証は未実施。
- 正式preflight：CPU 2.053秒、elapsed 2.054秒、最大RSS 117,640 KiB。上限CPU120秒・wall300秒・address space2GiB、1 thread。source inventoryは796ファイル、SHA `72332ec2eb69abfd562fb82f35b7d7db14fdc05a5f077f5e222cebb5837d5bc4`。
- 保護対象28,293ファイル、既存tracked18,246ファイル、既存venv内16,477ファイル/リンクを前後照合。既存3 Draft／18 assetsはread-only metadata一致。asset再取得・新たな復元試験はしていない。
- 元960 slots／934 valid／25 known failures／HTTP520欠測1（global942）、961 fixed-scope attempts／991 historical calls／unknown holds2／global940の承認済みretryを維持。Goal007–012と原台帳・authority・approval・checkpointを変更していない。
- 研究生成・model query・token-count・認証情報照会・judge API・人手実験・推論・download・installはすべて0。

保護は14 read-only locks、Python接続禁止とファイル書込みガード、原本hash照合による。OSレベル隔離の保証とはしない。公開成果物だけでprivate原本を復元できるとは主張しない。

再現入口は `offline_evaluation.analysis.goal013.run_preflight`。実行記録の引数は `--output-root /root/minecraft-runs/goal013-20261017T111808Z --code-commit 232963b46b9cbd975f77f9f0401e5f57b0e15cc9`。この既存runは再利用不可で、実行時HEADとcode/spec bytesの一致を検査する。実スコアや関連解析の実行器はない。

成果物：[source CSV](SOURCE_BINDING_180.csv)、[source JSON](SOURCE_BINDING_180.json)、[coverage](SOURCE_COVERAGE.json)、[source receipt](SOURCE_BINDING_RECEIPT.json)、[scorer監査](OFFLINE_SCORER_AUDIT.json)、[選択規則](MODEL_SELECTION_RULE.json)、[採点前規約](SCORING_PROTOCOL.md)、[未計算provenance](SCORE_PROVENANCE.json)、[尺度妥当性](SCORE_VALIDITY_JA.md)、[関連解析の未実行範囲](RQ2_EXPLORATORY_RESULTS_JA.md)、[研究判断](RESEARCH_DECISION_JA.md)、[検証](VALIDATION.json)、[保全](INTEGRITY.json)、[MANIFEST](MANIFEST.json)。

公開先は `research/server-only-20261013`。開始HEADは `74aaf90795ae17a22c4849156068e6c26c186f22`、mainは `7309327efd1c19dda793b19e8d8e6ce0b5868db9` を保全する。結果commit・remote HEAD・worktree cleanは最終push後に別途照合し、最終応答に記載する。
