Goal015C：schemaキー順序の局所検証と12呼出し診断の未承認準備

この仕様は実モデル計算前の固定である。親Goal015の72件を置換せず、Goal015Aの残8件を開始しない。Goal015Bで確定した非空carve-only48件、truncation22件、touch上限拒否2件を変更しない。原因は未確定である。

対象は既知正解の合成課題1種類、2 Builder、3条件、2予定反復。実建築数や独立な研究Nを増やす試験ではない。失敗後の再試行、結果に応じたモデル・prompt・cap・順序の変更、親実験への自動移行はない。

入力は全条件共通のJSON `{"summary": TASK_SUMMARY}`。TASK_SUMMARYは次の原文をそのまま使用し、既存cohort serializerでJSON化する。

> Starting from an empty world, return exactly three operations in this order: first, fill the inclusive cuboid from (0,0,0) to (1,1,1) with stone; second, carve only the single voxel at (0,0,0), using that same coordinate for both endpoints; third, set the voxel at (3,0,0) to glass. Leave every other coordinate empty. Do not add, omit, or repeat operations. Return only the operations JSON object.

system promptは `offline_evaluation/resmoke_v2/prompts/builder.txt` の凍結bytesそのまま、SHA256 `ba7a555860628c1ad55ba5b65200ac3cf0d528be86c42f46dbb5bbdd61e6ccac`。例示を追加しない。

```json
{"operations":[
 {"op":"fill","x1":0,"y1":0,"z1":0,"x2":1,"y2":1,"z2":1,"block":"stone"},
 {"op":"carve","x1":0,"y1":0,"z1":0,"x2":0,"y2":0,"z2":0},
 {"op":"set","x":3,"y":0,"z":0,"block":"glass"}
]}
```

正解は(0..1,0..1,0..1)から原点を除くstone7セルと(3,0,0)のglass1セル。primaryは元座標・literal材質での8セル完全一致。要求された3操作の順序・値への適合は別列とする。空白・JSON objectのキー順は採点条件ではない。旧executorの材質正規化後に一致しても、literal spellingを変えるalias修復が必要ならtask_passにしない。移動、回転、GT、prefixの補完、fence除去、JSON修復を使用しない。

| 条件 | native指定 | Aから許可する差分 |
| --- | --- | --- |
| A `NATIVE_CURRENT_ORDER` | 有効 | Goal015の送信前canonical cloneと同じ全objectのソート順 |
| B `NATIVE_ORIGINAL_PROPERTY_ORDER` | 有効 | 3 operation branchのproperties辞書の列挙順だけ元schemaへ戻す |
| C `TEXT_JSON_INSTRUCTION_ONLY` | 無効 | OpenAI text.format／Anthropic output_config.formatを削除し、空の親objectも削除 |

A/Bではrequired配列、anyOf配列、enum、型、説明、name、strictを変更しない。受理するJSONの集合は同じで、生成分布の同一性は仮定しない。CにはJSON mode、tools、回答例を追加しない。A/B/Cとも同じstrict validatorを使用する。

OpenAIは `gpt-5-mini-2025-08-07`、reasoning medium、temperature/top_p/seedなし、Responses endpoint、service_tier default。Anthropicは `claude-haiku-4-5-20251001`、temperature1、thinking disabled、Messages endpoint、service_tier standard_only。全条件でoutput cap32768、timeout600秒、画像・toolsなし、concurrency1、retry0。

| call | repeat | Builder | 条件 |
| --- | --- | --- | --- |
| goal015c_call_001 | 1 | OpenAI | A |
| goal015c_call_002 | 1 | Anthropic | C |
| goal015c_call_003 | 1 | OpenAI | B |
| goal015c_call_004 | 1 | Anthropic | A |
| goal015c_call_005 | 1 | OpenAI | C |
| goal015c_call_006 | 1 | Anthropic | B |
| goal015c_call_007 | 2 | Anthropic | B |
| goal015c_call_008 | 2 | OpenAI | C |
| goal015c_call_009 | 2 | Anthropic | A |
| goal015c_call_010 | 2 | OpenAI | B |
| goal015c_call_011 | 2 | Anthropic | C |
| goal015c_call_012 | 2 | OpenAI | A |

反復は事前に予定した別試行でありretryではない。この順序でprovider間時間差を完全に除去したとは主張しない。

**Wireと契約。** 最終送信body6種類を固定し、12 schedule行がそれらの絶対path・bytes・SHAを参照する。通常JSON値hash、object順序を保持した表現hash、wire SHAを区別する。本番adapterは検証したbytesそのものをRequest.dataへ渡す。送信直前にhash・bytes・一致・公式hostをdurable ledgerへ保存してからHTTPへ進む。headersと資格情報は保存しない。過去wireがない旧O2例は凍結コードからの再構成であり、remote受信証明ではない。

**費用。** 新規上限USD3.25は未承認の提案。公開文書の標準単価を使い、OpenAIは入力400000×0.25/1M＋出力32768×2/1M＝USD0.165536、Anthropicは入力200000×1/1M＋出力32768×5/1M＝USD0.363840を各6件予約する。合計最大USD3.176256。OpenAI文書の最大入力272000より大きい400000を保守予約に維持し、これを送信可能入力数とは扱わない。実費予測ではなく、cache割引を使用しない予約上限である。親既知費用USD4.43765700、旧USD5、未承認提案USD7と別枠。

公開本文のGET、取得URL/実UTC/hash、明示的読取りreview、sealを別に保存する。captureだけでreviewedにしない。最初の実GET開始から24時間有効。同契約の新reviewed証拠を `--pricing-evidence` で渡せるが、plan ID・ledger・予算をresetしない。現行資料が契約・cap・単価と異なる場合、同計画へのsealを拒否する。snapshotの公表終了日以後も起動を拒否する。account利用可否や内部実装は照会しない。

**将来の起動。** canonical launcherは未開始・実装・wire・親子状態・lock・料金期限をキー入力前に検証する。本番planはapi_allowed=false/budget_approved=falseのまま、利用者の実TTYの明示承認後に別approval/authority/ledgerを新規作成する。承認文は金額・plan IDを共通関数で表示し、別資料に推測した文字列を転載しない。コピー起動・既存有効化・二重実行を拒否する。

各送信前に全額予約、送信後に取得できたusageだけ計上する。不明usageはholdを残し、同IDを再送しない。モデル出力失敗では残りの指定条件を継続し、通信・認証・モデル契約・予算等の運用エラーでは停止する。再開は明示コマンドと既存authority確認の上でNOT_SENTだけを対象とする。進捗と終了status JSONを保存し、exit0だけを完了根拠にしない。

**評価。** 完了rawから独立抽出し、strict JSON/schema検証、旧executorと独立疎辞書executorを照合する。HTTP/finish/usage、JSON有効性、操作件数と各種別件数、三操作適合、final voxel数、grid完全一致、truncation/touch拒否を記録。正常empty／carve-onlyはprotocol-validでもtask_pass=false。known output failure・UNKNOWN・NOT_SENT・CPU評価不能を分離し、不明値を0へ変換しない。12予定行を残し、条件ごとの実送信・応答完了・settlement・protocol-valid・厳密成功・費用を区別する。

判定は計算前に次へ固定する。2反復から機構・一般化・統計的優位性を証明しない。

| 観測 | 許される解釈 |
| --- | --- |
| A失敗、B成功 | このミニ課題でproperty orderingが関与する候補を強める。旧72件の原因やprovider bugを確定しない |
| A/B失敗、C成功 | native format指定を含む経路への依存候補。provider追加promptも異なり得るためconstraint decoding単独の効果ではない |
| A/B/C成功 | 現時点でこの課題の基本動作が可能。実建築の解消は保証しない |
| 全失敗／混在 | 保存traceから次の限定仮説を判断する。自動retry・モデル変更・別課題への移行はしない |

bootstrap・p値・新ランキングを作らない。公開仕様は現在の構造・順序の説明であり、過去carve-onlyの原因認定ではない。[OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)、[Claude structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)、[JSON Schema anyOf](https://json-schema.org/understanding-json-schema/reference/combining)。

**保存。** finishは12予定行の評価、専用checkpoint、新しい未公開Draft作成、再取得SHA検証、安全な別directoryへの復元を行う。生成とsave-only retryを分離する。既存Draft/assetを変更しない。Python環境は外部依存として版を記録し、wire・task・prompt・code・料金証拠・controls/raw/evaluationを保存対象にする。履歴provenance内部の全旧実験参照を再帰展開せず、外部依存を明示する。

今回実施するのはCPU/synthetic/実PTY fixtureとmock保存テストのみ。本番launcher、研究API、資格情報照会、実承認、残8件、実Draft操作は0。旧全suiteや数万ファイルの再監査は行わず、読む必要のある凍結ファイルの開始/終了hashを照合する。既存Pythonを使用しinstallやOS設定変更はしない。
