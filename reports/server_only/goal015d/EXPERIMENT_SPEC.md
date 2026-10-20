Goal015D：ミニ課題の結果確定と実建築2件・A/B16呼出しの未承認準備

この仕様は実建築canaryの実モデル計算前に固定する。Goal015Cの保存済み12件は再生成・再採点せず、正式集計・原本hash・call対応を確認した。OpenAIのtask_passはA0/2、B2/2、C2/2、AnthropicはA0/2、B2/2、C0/2。Anthropic Cの002/011はMarkdown fenceを含む本文のstrict JSON失敗のまま。fenceを除去した内部を再評価しない。Goal015Cの既知実費はUSD0.01865750である。

当該合成課題・設定・2反復ではproperties順の介入に伴う成否の切替を観測した。独立課題は1種類。旧72件の原因、provider内部実装、op先頭だけの効果、一般的な100%成功やモデル間優劣は確定していない。Bは座標keys等の順も復元する。次の試験も順序介入全体の局所診断であり、SUMMARY_ONLY対FULL_RECORD本実験ではない。

**固定する対象と入力**

親Goal015 scheduleの先頭から重複を除いた最初の2建築、`buildings_100_v4_building_082`、`buildings_100_v4_building_003`の順を使用する。既存結果に応じた選別ではなく、利用済み建築によるdevelopment canaryである。未見holdoutとは呼ばない。

入力は親FULL_RECORDのUTF-8 bytesをそのまま専用inputs directoryへ複製する。親両Builderの保存payloadにあるinput stringともbyte一致することを検査する。summary＋7フィールド、数値、配列順、材質の曖昧さを維持する。metadataを戻さず、summary-only、GT寸法・palette・正解、追加例示を入力に加えない。欠落・不一致なら対象差替えをせず停止する。

| scene | bytes | FULL input SHA256 |
| --- | ---: | --- |
| building_082 | 2568 | a12a7a9cc16a4af90cecfff3fd88ebe96c6a5b567112623a7c2470a29ac610e6 |
| building_003 | 2851 | a37fd73c5ac995cf0be5f834d258af076a65b87917dbb0c24349271a68f08a96 |

凍結builder system promptは `offline_evaluation/resmoke_v2/prompts/builder.txt`、SHA256 `ba7a555860628c1ad55ba5b65200ac3cf0d528be86c42f46dbb5bbdd61e6ccac`。元のbytesを共通利用する。

**条件と固定順**

A=`NATIVE_CURRENT_ORDER` はGoal015CのAと同じソート後schema。B=`NATIVE_ORIGINAL_PROPERTY_ORDER` はAの3 branchのproperties辞書の列挙順だけ次へ戻す。

- set：`op,x,y,z,block`
- fill：`op,x1,y1,z1,x2,y2,z2,block`
- carve：`op,x1,y1,z1,x2,y2,z2`

required/anyOf配列、enum、型、name、strict、description、追加制約は同一。受理集合は同じでも生成分布の同一性を仮定しない。minItems、最初の操作強制、carve禁止、追加prompt/回答例、Cの再試行を入れない。

OpenAIは `gpt-5-mini-2025-08-07`、reasoning medium、temperature/top_p/seed省略、Responses、default tier。Anthropicは `claude-haiku-4-5-20251001`、temperature1、thinking disabled、top_p/top_k/seed省略、Messages、standard_only tier。双方native output、output cap32768、timeout600秒、streamingなし、画像/toolsなし。alias/fallback、新SDK、cache write、tier変更はしない。

| call ID末尾 | scene | Builder | 条件 | 反復 | 隣接pair |
| --- | --- | --- | --- | ---: | ---: |
| 001 | 082 | OpenAI | A | 1 | 1 |
| 002 | 082 | OpenAI | B | 1 | 1 |
| 003 | 082 | Anthropic | B | 1 | 2 |
| 004 | 082 | Anthropic | A | 1 | 2 |
| 005 | 082 | Anthropic | A | 2 | 3 |
| 006 | 082 | Anthropic | B | 2 | 3 |
| 007 | 082 | OpenAI | B | 2 | 4 |
| 008 | 082 | OpenAI | A | 2 | 4 |
| 009 | 003 | Anthropic | A | 1 | 5 |
| 010 | 003 | Anthropic | B | 1 | 5 |
| 011 | 003 | OpenAI | B | 1 | 6 |
| 012 | 003 | OpenAI | A | 1 | 6 |
| 013 | 003 | OpenAI | A | 2 | 7 |
| 014 | 003 | OpenAI | B | 2 | 7 |
| 015 | 003 | Anthropic | B | 2 | 8 |
| 016 | 003 | Anthropic | A | 2 | 8 |

ID prefixは `goal015d_call_`。8隣接ペア、16予定生成、concurrency1、retry0。両条件とも同時期に新生成する。旧A72件や旧非空Oを対照へ代入しない。反復は失敗retryではなく、成績を見た順序変更や追加反復はない。

**送信と独立予算**

新8 wire（2scene×2Builder×2条件）を固定し、16行から参照する。FULL inputとprompt、型・値tree、ordered representation、body SHAを分離して束縛する。Bを送信前に再ソートすると拒否。新adapterは検証済みbytesそのものをRequest.dataへ設定し、その長さ・hash・一致・公式hostをdurable保存してからPOSTする。headers/キーを保存しない。この証拠はローカル送信直前のもので、remote受信の独立証明ではない。

新提案上限は **USD5.00、未承認**。料金は現時点で有効なGoal015Cの取得本文・明示review・hash束縛をそのまま検査して再利用する。新A/B契約は同じprovider条件のnative subsetであり、Cのtext条件を含めない。元の取得UTCを変えない。

| Builder | full-context入力予約 | output cap | 入力/出力単価 USD/百万 | 1件予約 | 8件予約 |
| --- | ---: | ---: | --- | ---: | ---: |
| OpenAI | 400000 | 32768 | 0.25 / 2 | 0.165536 | 1.324288 |
| Anthropic | 200000 | 32768 | 1 / 5 | 0.363840 | 2.910720 |

合計最大予約は **USD4.235008**。実費予測ではなく、入力全context＋output capの保守予約。400000を実送信可能な最大入力数とは呼ばず、実際に入力/出力上限が同時達成可能とも仮定しない。cache割引を使わない。親USD4.43765700、015C実費USD0.01865750、旧USD3.25上限、未承認USD7.00枠、旧USD1.10外挿を流用しない。

証拠期限は最初の実GET開始から24時間、今回 **2026-10-21T10:00:58.172322Z**。同契約の新しい明示reviewed evidenceを `--pricing-evidence` で渡せる。plan複製・予算resetはしない。期限切れやモデル/単価/契約の変更は送信前に停止する。既存証拠を期限内再利用できる今回は新規公式GETも行わない。アカウント可用性、model-list、token-count、認証APIは未照会。

**将来の起動と停止**

本番planはAPI無効・未承認、実approval/authority/ledger未作成で固定する。canonical launcherは未開始、committed code/plan/input/wire、料金証拠、親Goal015/旧補完Goal015A/完了Goal015Cとの競合を、キー入力前に確認する。不足キーだけ非表示入力し、再度preflightする。TTY read/writeは別ハンドル。承認文はplan IDとUSD5.00を単一関数から表示し、利用者が実TTYで手入力する。script/Codexが承認を入力・pipeしない。

実行時は各隣接A/Bペアの両方の予約を最初のPOST前に一度のdurable ledger更新で確保する。既知usageだけ精算し、unknownと予約済みNOT_SENT partnerを区別してholdを維持する。HTTP/認証/通信不明/契約不一致は停止し、同callを自動再送しない。明示resumeが許される場合も確認済みauthorityのNOT_SENTだけ。成績による方法変更はない。known output failure、正常empty、carve-onlyは結果として残し、予定16件を続ける。

前後の進捗はn/16、scene、pair、condition、state、known費用、holdを表示する。終了status JSONと保存証明を確認し、exit0や16 SETTLEDを正確な建築成功と同一視しない。親/旧補完や本実験への自動移行はない。

**評価と解釈**

実建築にGoal015Cの8セルtask_passを流用しない。保存raw→独立text抽出→strict validator→凍結executor/独立疎辞書executorを照合し、response/usage、known output failure、非empty、操作種別件数、write有無、carve-only、truncation、2M touch拒否、最終voxel数を16行へ記録する。正常emptyはprotocol-valid。未送信、運用不明、出力失敗、CPU不能のgrid/metricを捏造しない。fence除去、途中JSON修復、carve→fill、GT-informed material repairを行わない。

GTは評価時だけloadする。既定のaligned occupancy IoU、surface F1@1、precision/recall、dimension MAE、material-aware IoUを凍結Goal015評価器で計算する。occupancyで選んだ一つの最適shiftを全指標で共有し、新しいalignment探索法・指標を追加しない。非空は正確な再建築の証明ではない。

16行全件、2建築×2Builder×2条件の8セル（各2予定）、8隣接ペアを保存する。各セルで予定/送信可能/POST返却/完了/usage精算/valid/nonempty/carve-only/失敗と費用を区別する。B−Aは両側共通に定義された指標だけで計算し、NULLと理由を保持する。failure0補完、bootstrap、p値、成功率ランキングは作らない。独立建築は2件。

| 観測 | 許される解釈 |
| --- | --- |
| Aがcarve-only/失敗、Bが非空 | 当該建築でも順序依存の現象を支持する観測。正確さはGT指標で別に見る。旧72件の反実仮想は回復できない |
| A/Bとも非空 | ミニ課題と同じ差は当該建築で再現しない |
| A/Bともcarve-only/失敗 | 単純な順序復元だけでは実建築への対処が不十分 |
| 混在 | 全結果を保持し、都合の良いセルだけで結論づけない |

provider cache/decoder内部のbugは断定しない。op位置だけ、座標順だけの効果は未分離。終了後はBを使う新S/Fプロトコルを検討できるか人間へ報告するだけ。旧残8件をBで補完せず、旧72件を上書きせず、80件を自動実行しない。

**保存と準備検証**

将来の16件実行後は評価→専用checkpoint→新規未公開Draft→再取得SHA照合・安全復元へ進む。途中結果もNULLを保持して保存できるが、partialを完了としない。保存失敗は同checkpointのfinish/save-onlyで回復し、生成を繰り返さない。既存Goal015C Draft418290999/asset659057302へ変更・公開・削除を行わない。

archiveに必要なplan/controls/input/wire/raw/evaluation/code/spec/料金資料/GTファイルを束縛し、Python環境等の外部依存を明記する。archive単独で全履歴・環境を完全再現できるとは主張しない。

今回の検証は新差分fixtureのみ。A/B差分、FULL一致、B再ソート拒否、mixed set/fill/carve、empty/非空/失敗、8wire×2反復、ペア予約、UNKNOWN hold、二重起動、実PTY拒否/承認、mock保存復元・save-onlyを検証する。本番launcher・承認・研究APIを使わず、旧全suite、全raw再解析、install、OS/SSH/Codex設定変更をしない。必要な旧sourceと新依存だけ開始/終了hashを確認し、新規コード/仕様/小表/wire/未承認plan参照を通常commit/pushする。
