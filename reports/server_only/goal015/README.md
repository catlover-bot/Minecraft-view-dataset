Goal015は **PILOT80_READY_AWAITING_EXPLICIT_APPROVAL**。実装、未承認plan、入力・順序・費用の固定、CPU/fixture検証を完了した。研究API送信0、キー照会0、モデル照会0、アカウントavailabilityはNOT_RUN。実planのapproval/authority/ledger/enabledは未作成で、新USD5は未承認の提案である。

対象は保存OpenAI説明の10建築。提示順は `buildings_100_v4_building_082, 003, 092, 025, 042, 052, 013, 024, 033, 047`（省略部分も同じprefix）。保存5サイズ層から各2件を、選択前commit `9c32add47cb724b1afc01924aa047cbd5619258a` のhash規則だけで選んだ。過去score・失敗・例示sceneは使わず、入力欠損による置換0。全IDと原本hashは [source_binding_10.json](source_binding_10.json)。

SUMMARY_ONLYは元summaryだけのJSON、FULL_RECORDは同じsummaryとbuilding_type/shape/dimensions_estimate/materials/elements/rebuild_hints/uncertaintiesのJSON。同じ型保存serializationで、原文字列・Decimal・配列順・曖昧さを保持した。provider/model/building/created_atは双方から除外し、未知top-level fields0。S/Fのsummary SHAは全10件一致する。Fは旧O raw replayではない。追加7フィールドに伴う長さ・重複・矛盾も変わるため、純粋な情報量やprose/IR効果とは呼ばない。

10×2 Builder×2条件×2反復＝**80予定呼出し、40隣接paired blocks**。repeat2でS/F順とBuilder順を反転し、各層1件ずつ2巡する。concurrency1、retry0。全scheduleと40固有payloadのSHAは [schedule_80.json](schedule_80.json)。原文payloadはcanonical private runだけに保存し、Gitへ追加していない。

| Builder | request/response model | sampling / reasoning | output cap |
| --- | --- | --- | ---: |
| OpenAI | gpt-5-mini-2025-08-07 | medium; temperature/top_p/seed OMITTED | 32768 |
| Anthropic | claude-haiku-4-5-20251001 | temperature1; thinking disabled | 32768 |

凍結promptとnative operations schemaを共用し、新adapterだけを追加した。実応答を見て設定変更しない。公式文書GETをこのサーバーで再取得し、GPT5miniはdeprecated／2026-12-11停止予定、Haiku4.5はActive (legacy)と確認した。自動fallbackはない。各送信前に公式資料が24時間以内であることを要求する。詳細は [PRICING_EVIDENCE.json](PRICING_EVIDENCE.json)。

費用は、全991旧試行を対象にした該当known usage128件の参考外挿が **USD1.098149219**。新S/Fの出力長は未知で、予測保証ではない。full-context input＋32768 outputの全80最大予約は **USD21.175040**。paired blockはOpenAI0.331072、Anthropic0.727680。提案USD5は最悪予約より16.175040少なく、途中停止し得る。最悪条件を覆う推奨額21.18は参考提案だけで、plan上限を変更していない。[COST_ESTIMATE.md](COST_ESTIMATE.md) にpayload byte数、tokenizer不在、全履歴と旧予算参照を分離した。

新規テスト **162 PASS / failure0 / error0 / skip0**。実PTYの完全一致承認・拒否・Enter・EOF・nonTTY・read failure・二重起動、paired予算／unknown holds／再送禁止、copy防止、価格更新、fake Draft ID再利用・asset ID復元・保存専用retryを検証した。新しい全suiteはCPU37.30秒、経過45.41秒、最大RSS84,004KiB。過去の全テストやCLIP再採点は行っていない。

mockの正常80件と、最初のtransport不明→NOT_SENTだけresumeした80件の2シナリオが完走した。後者は79有効＋1unknownでNULLを維持し、同じcallを再送していない。各応答・parsed/voxel/metricsとlocal checkpoint復元のhashを照合した。mock160呼出しを研究API160回や実測160成功とは扱わない。[MOCK_END_TO_END_RECEIPT.json](MOCK_END_TO_END_RECEIPT.json) を参照。

実canonical planでpreview/statusと未送信のevaluate/aggregateを確認した。全80行NOT_SENT、実計測0、GT voxel読取0、研究API0であり、承認・台帳・lockを作っていない。評価は旧strict translationとGoal010v2定義を再利用し、IoUとsurface F1@1を同じoccupancy shiftで測る。utilityとcommon-validの両paired差をscene等重みで報告し、独立Nは最大10。正常emptyと既知失敗0、運用NULLを区別する。詳細は [EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md)。

旧保護対象28,432 files、28 trees、tracked18,376とmainは終了確認で不変だった。旧960/934/25/HTTP520欠測1、961 attempts、991 calls、unknown holds2、global940承認retry、旧予算／承認／3Draftを変更していない。開始検証は直前の保存inventory検証を再利用しGoal014追加分を拡張、終了全照合を1回行った。原本確認資源は [INTEGRITY_BEFORE.json](INTEGRITY_BEFORE.json)／[INTEGRITY_AFTER.json](INTEGRITY_AFTER.json) に分離した。

canonical rootは `/root/minecraft-runs/goal015-20261017T170546Z`。plan IDは `5b80fdef342dd2b9f68b3c2bec2d3d83af2731c3b5a9c8940959846ea11c68ce`、実plan SHAは `6533090e07090b7279f0383937631c609b17c84c922f8cd8deadcd2b9ea22257`。55のcode/promptファイルとspec/payload/価格契約を束縛し、Git内のcopyから実行できない。[EXECUTION_BINDING.json](EXECUTION_BINDING.json) が実原本を指す。

利用者が後で開始する実在コマンドと価格再確認・resume・evaluate・保存専用retryは [OPERATIONS.md](OPERATIONS.md)。実TTYへの承認入力と有料生成は別実行である。小規模development pilotであり、十分なpower、人間品質妥当性、独立外部検証、各フィールドの原因を確定しない。

専用branchへnon-force commit/pushしてremote一致とcleanを確認し、成果commitの実SHAは最終応答とcanonical PUBLICATION_RECEIPT.jsonに記録する。mainは `7309327efd1c19dda793b19e8d8e6ce0b5868db9` のまま。
