# Goal002 — CPU検証と課金開始準備

研究APIは **NOT_RUN・送信0回**。今回の実装とCPU検証は完了し、smokeの課金承認を待つ。新実験の生成完了とアーカイブの保存完全性は別に記録する。旧ARRの原本不足は新実験の停止理由にしない。

実行場所は `88.216.191.99 / Ubuntu24.04.5 / root / /root/workspace/Minecraft-view-dataset`。開始HEADは `5dc0eb22be331b3883c47496a2d4856456ba5697`、専用branchは `research/server-only-20261013`。現在の実効sandboxは `workspace-write`、承認審査は `auto_review`。通常コマンドが起動することを実測し、追加run/checkpoint書込みとGitHub操作は対象限定の審査を通す。`hostname -I` は通常sandboxでは `Operation not permitted`、対象限定の承認後に指定サーバーIPを確認できた。これはsandbox起動エラーではなく、OS隔離の保証もしない。root接続を維持し、OS/SSH/ユーザーを変更していない。

新runは `/root/minecraft-runs/goal002-20261013-141854`。最終CPU結果は `final_cpu/`、旧仕様や途中確認の結果は保全したまま。新checkpointは `/root/minecraft-checkpoints/goal002-20261013-141854`。保存・復元の確定結果とSHAは後続の `checkpoint_index.json` に記録する。

| 対象 | 現在の状態 |
|---|---|
| historical_arr_replay | BLOCKED_MISSING_ORIGINALS。過去のsource_reference/manifestを再利用し、重複探索しない |
| legacy_i2t2b_v5_reevaluation | new_implementationの診断。旧ARR1200と異なる版 |
| server_v2_cpu_controls | 実200 GTでPASS。LLM Builder成功に数えない |
| new_v4_two_builder_experiment | PLANNED_NOT_RUN。新第一Builderの対照結果も未生成 |
| experiment_completion | NOT_RUN |
| archive/restored integrity | checkpoint索引で個別に実測状態を報告 |

## 評価仕様と実測

主評価は `server_v2.2_strict_translation`。有限のoccupied集合に対して、疎な相対座標差または整数NTTで全XYZ平行移動の最大overlapを厳密に求める。浮動小数FFTの丸めは用いない。tieはL1最小、XYZ辞書順。材質とrepairもoccupancyで選んだ同じshiftを使用する。bounded/radius1は旧版と比較する別列で保全し、資源上限時は `ALIGNMENT_RESOURCE_LIMIT`。回転・scale・GTによる出力補修は加えない。

非対称・tie・空集合・材質不一致・遠い原点を含む独立総当たり照合に加え、実200 GTでExact-spec往復200/200、Executor control200/200、既知shift3種類で600/600、遠い原点200/200。worker2/BLAS1、68.02秒、最大RSS108436KiB。[CPU結果](cpu_controls.json)を参照。

保存された1200出力は全て原本voxel/bbox/actionsのSHAを保持し、geometry1200、材質1162、材質UNKNOWN38、影響14建築。旧32件は引き続きUNKNOWN。追加6件はU32切詰め2件と構文上有効な複数候補ラベル4件で、GT穴埋めは行っていない。確定切詰めは計4件、再構成データ0件。これは読込み・材質解釈の診断であり過去のモデル/Minecraft実行失敗とは断定しない。全6条件のcommon material-validは186建築、条件対ごとの分母はJSONに保存する。

厳密位置合わせは1200/1200完了、資源上限0。radius1よりIoUが高い出力は1144件。worker2/BLAS1、206.69秒、最大RSS248652KiB。[比較CSV](alignment_comparison.csv)、[条件別・common-valid結果](legacy_geometry_summary.json)、[38件診断](material_diagnostics.json)、[旧32件との差分](material_diagnosis_delta.json)を参照。最終処理の原本・評価コードhashは開始終了で一致した。

## コホートと入力

v4の100建築について3説明モデルとGT/画像参照を原本SHAに照合。GT occupied数のrank五分位、同数ならscene ID順、seed20261013、SHA256 counter/rejection Fisher–Yatesの固定仕様で10 smoke・60 evaluation・30 restを選ぶ。精度や材質診断を選抜に使わない。cohort hashは `eeed4b62047106f4ce75d853c4e33dc22cac812224282ceca9d6b84f6b696f63`。[固定protocol](protocol.json)に全IDと入力hashを記録。

各Builderの本評価はO180/J180/P60/E60、480件、両者960件の計画。初回smokeは固定小/大側のbuilding060/075、保存済みOpenAI説明のO/JとP/Eで各8、合計16。全件prepared、入力上限による除外0。CPUで準備可能なことは、実APIでのcontext受理や16384出力内の完了を保証しない。

O/Jのrequestは保存済み説明と公開共通promptのみ。Jは重複キーを拒否し、型付きJSON木・数値・配列順・null・不確実性を保持する新しい再シリアライズで、旧ARRのcanonicalizationではない。両providerのserializerでGT等のsentinel漏洩を検査した。P/Eは不可視内部も含むGT診断条件であり、Eはpalette/Z-runの読み取り・展開能力も要求する。絶対原点が画像/説明から不明な場合があるため、absolute誤差を説明能力だけに帰属させない。

建築単位のpaired集計・cluster bootstrapと3つの事前宣言Holm familyを使用する。Oracleは説明数で複製しない。失敗0は識別された完了requestのparse/validation/execution/truncationに限定し、未実行・認証・unknown usage・予算停止を除外。MAE/repairはnullとcommon-valid分母を使用。smokeには推測統計を付けない。全計画行を集計に残し、同一providerの重複runを拒否する。

## 検証と課金準備

新server_v2スイート242 pass、fail/skip/error0、pytest15.79秒、プロセス16.20秒、最大RSS88500KiB。既存venvの `pip check` はPASS、依存取得なし。旧6 collection errorsは元ログ/理由/hashを新runの `previous_legacy_suite/` に保存し、ダミー実装や一括skipで消していない。[テスト索引](new_tests_summary.json)。開始時の空き容量は約80GiB。

固定モデルはOpenAI `gpt-5-mini` とAnthropic `claude-haiku-4-5-20251001`、fallbackなし。公式料金/context/取得日/response aliasを [official_pricing.json](official_pricing.json) に保存。アカウントでの利用可否はNOT_RUN。公式にはOpenAI対象モデルはDeprecatedと表示され、利用不可ならその事実を記録して停止する。OpenAIの思考と可視出力は同じcap/課金に含まれ、Anthropic payloadはextended thinkingを有効にしていない。

初回smokeの提案上限は両provider合算20送信、再試行4、入力あたり2attempt、5USD、token8327680、output16384、concurrency1、隠れたretry0。永続ledgerとlockで予約を送信前に保存し、unknown usageは予約保持＋停止。承認はtargets/config/input/prompt/source/予算と将来の新enabledディレクトリに固定する。現在はdisabledとmanual_confirmed=falseのみ。実際のキー存在bool、準備identity、保存SHAは後続索引に記録する。

実在CLIに基づく [操作手順](OPERATIONS.md) を参照。有料生成コマンドは今回未実行で、別途利用者の対象・費用・上限の明示承認が必要。本評価960件へ自動で進む機能はない。

保存・復元を完了した。[新checkpoint索引](checkpoint_index.md) にDraft411807868、全assetのhash、source再取得、復元範囲を記録した。API送信0回、キー存在boolは両provider false。新v4準備物の完全性と旧研究全体のPARTIALを区別する。CPU wall/RSSは並行実行した各worker2/BLAS1プロセスの検証値であり、単独実行の速度比較ではない。
