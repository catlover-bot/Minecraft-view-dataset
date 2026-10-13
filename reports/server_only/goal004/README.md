# Goal004 — native structured outputとcuboid Eの再smoke準備

状態は **PROTOCOL_READY_AWAITING_EXPLICIT_RESMOKE_APPROVAL**。追加研究API送信0回。固定16入力のdisabled計画と、利用者が一度だけ明示承認する実在ランチャーを実装した。実アカウントでのschema受理・P空出力の解消・E出力の成功はNOT_RUNで、CPU検証と区別する。

新実装は `offline_evaluation/resmoke_v2/`。旧 `server_v2` のhash-bound生成ファイル・旧入力・prompt・config・state・ledger・attempt・raw・評価・承認は変更していない。旧validatorで旧campaignが読めることをテストし、Goal003時点の69証拠ファイル＋4制御ファイルとのSHA一致を確認した。[原本照合](originals_verification.json)。開始HEADは `7e2530e19ba783cab44eec47e4cca25ab695a69a`、コードcommitは `d5d3e750423b9f900654097a65bde62bf230bf4f`、ブランチは `research/server-only-20261013`。

実効設定はworkspace-write / auto_review。root・既存worktreeを維持し、追加出力・Git・公開依存取得は対象を限定した審査で実行した。通常コマンド成功はOSレベルの隔離保証ではない。`hostname -I` 単体は `Operation not permitted` で取得できなかったが、sandboxの起動と通常コマンド・CPUテストは成功した。[実行条件](execution_context.json)。

共通schemaはtop-level operationsだけ、set/fill/carveのanyOf、全必須キー、additionalProperties=false。carveにblockは存在せず、integer・重複キー・大文字材質・余剰キー・フェンスを補修しない。OpenAI Responsesのtext.formatとAnthropic Messagesのoutput_config.formatに同じschemaを渡す。共通のstrict parser・Validator・Executor・strict translation alignment・集計を再利用した。[公式API/モデル確認](official_support.md)、[新料金記録](official_pricing.json)。

Pは **GT-attributes-only diagnostic**。旧Pの寸法・bbox・材質集合・voxel数だけの本文を維持し、OやGTの部品形状を追加しない。共通promptは全条件について「与えられた制約に従う一つの案を作り、不足する配置は妥当な案を選ぶ」と明示する。空operationsや全carveは構文上許すが、非空再建築と分離し、実測のIoU0と定義済みMAE/repairを保持する。NOT_RUN・unknown usageは0へ変換しない。

Eは同材質・非重複cuboidによる完全な離散仕様。最初の残存セルをY,X,Z順で選び、Z→X→Yの順に同材質セルだけを拡張する決定的分割で、最小分割を主張しない。inclusive XYZ、world/localとorigin、palette、省略airを明記。GTからactionを作る参照実行はCPU controlであり、LLM成功に数えない。

| 固定GT | 旧Z-run→新cuboid | 旧→新E入力bytes | 旧→新参照action bytes | 新参照tokens |
|---|---:|---:|---:|---:|
| 060 | 1005→334 | 15007→7943 | 81280→26943 | 13224 |
| 075 | 1426→657 | 21462→14583 | 114705→52576 | 25959 |

既存200GTすべてで往復・参照Executorが完全一致、36.679223秒。tiktoken0.14.0/o200k_baseのローカルtext計測を使用し、課金token-count APIは呼んでいない。075の参照列は旧16384を超えるため、**両provider・全条件32768**を未承認案とした。075では6809tokensの余裕があるが、最短出力やreasoning込みのモデル出力長は保証しない。Anthropicの正確なtoken数は不明。[全CPU結果](cuboid_cpu.json)、[CSV](cuboid_cpu.csv)、[取得した一時tokenizerの版とSHA](tokenizer_environment.json)。CSVのみCRLFをLFに正規化し、値は変更していない。

新計画は060/075 × O/J/P/E × 2Builder =16件、8つの意味入力を両providerへ渡す。O/J/P本文SHAは旧smokeと一致。新prompt/schema/Eにより新input IDとprotocol IDを作り、旧成功は流用しない。順序は建築→O/J/P/E→OpenAI/Anthropic。各入力最大1送信・追加retry0。既知usageの形式失敗・空出力・切詰めは次の未試行入力へ進む。認証・課金・モデル/schema非対応・不明usage・予算不足は停止する。

計画identity SHAは `047028e5645a90a94de70137d4b9ab579c9e56a08e583d723654276d2ac2cd77`、planファイルSHAは `d13b406df1bdd5f2fee5ca5821691b4fcb9c06875da4d2120d24104f45d2b74a`。実ファイルは `/root/minecraft-runs/goal004-resmoke-20261013T175539Z/plan.json`、`api_allowed=false`。[protocolと全対象hash](protocol.json)、[保存用disabled plan](disabled_plan.json)。

既存消費は14送信・retry4・173509tokens・USD0.25857425。追加枠16送信・retry0、累計30送信・retry4、累積USD上限5。追加token予約5324288、累計5497797。全16件最大予約と旧消費の合計はUSD4.49358225。親ledger/state/hashと固定authorityを照合し、別runやprovider単独の予算再初期化を拒否する。未知usageや中断済み送信は予約を保持し、自動再送しない。

既存307テストは一度実行して307PASS（pytest15.53秒、実時間15.84秒、最大RSS88684KiB）。新実装は104PASS（6.10秒）の後、実データ準備で検出した料金確認日validator互換性を修正し、関連20PASS（0.26秒、うち新規1件）。合計412個の異なるテストがPASS。旧6collection errorsは既存証跡を維持し、修復やskip追加で消していない。テストはsocket接続禁止で実行。symlink・台帳巻戻し・unknown usage・中断再開・親変更・snapshot・空出力・public保存境界を含む。[測定結果](test_results.json)。

開始コマンドと全承認範囲は[OPERATIONS.md](OPERATIONS.md)。実ランチャーのpreviewを別cwd・通信禁止で検証した。公式確認日は2026-10-13 UTCで、日付が変われば送信前に停止する。旧smokeとの改善差の因果解釈、smokeの有意差、本評価960の自動実行はない。

[新Draft Release 411963971](https://github.com/catlover-bot/Minecraft-view-dataset/releases/tag/untagged-0ac98cc3c92ab7b062b0)を未公開のまま保存した。対象Git commitは `104f87f13eff4c23489c598a26f4770f2e5c005b`。新準備物4ファイルだけのarchiveは19694bytes、asset `637834075`、SHA `faf2997e90d9b20fb3444615d80e65e2cc8e094ae9450049f7000115bcf8d8c5`。8assetすべてを別ディレクトリへ再取得し、GitHub digest・元・取得物のSHAを照合した。安全展開したplanは元と完全一致し、コピー先からの有効化は拒否された。旧14応答は再アーカイブせず、大容量sourceも既存assetのdigestを確認して参照を再利用した。[保存索引](checkpoint_index.json)、[7assetと展開の実測証跡](github_verification.json)、[8番目の証跡asset照合](verification_sidecar.json)。保存の完全性と新実験NOT_RUN、旧smoke PARTIALは別である。
