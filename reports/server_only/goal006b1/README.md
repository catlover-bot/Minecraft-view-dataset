# Goal006B1：実 TTY での追加承認入力を修正

Goal006B の承認画面で `/dev/tty` を `r+` で開くと、このサーバーの Python では `io.UnsupportedOperation: File or stream is not seekable.` になった。Goal006B1 はこの承認入力だけを修正し、CPU/fixture 検証と新しい未承認 plan の準備を完了した。**今回の有料研究 API 送信は 0 回**。API キーの値は取得・表示・保存せず、新しいキーの有効性を確認する API probe も行っていない。

現在の実効設定は `workspace-write` / `auto_review`。通常のコマンドと workspace 書込みは成功した。`hostname -I` は通常権限では `Operation not permitted`、対象を限定した承認後は成功した。[host_state.json](host_state.json) に現在の設定・uid 0・操作結果を記録した。これは観測した設定と操作結果であり、OS レベルの隔離を保証するものではない。root 接続、OS、SSH、ユーザー、既存作業場所を維持する。

## 原因と変更範囲

[tty_reproduction.json](tty_reproduction.json) は 2026-10-15 20:10:17 UTC の実 PTY による再現記録。controlling TTY と `stdin.isatty()` がある状態でも、Python 3.12.3 で `open('/dev/tty', 'r+')` が上記の例外を返した。同じ TTY を読取り用と書込み用の独立したハンドルで開くと成功し、両方の `isatty()` が true になった。この診断では launcher を起動せず、承認入力も送っていない。

production code の変更は [cli.py の `tty_approval`](../../../offline_evaluation/operations/goal006b/cli.py) に限定する。[独立レビュー](code_review.json) でも、それ以外の AST、planning/runner/checkpoint の bytes、元の凍結コード 39 files、旧 Goal006B report 20 files が不変であることを確認した。

テスト変更は [test_cli.py](../../../offline_evaluation/operations/goal006b/tests/test_cli.py) と新規 [test_tty.py](../../../offline_evaluation/operations/goal006b/tests/test_tty.py)。その他の追加ファイルはこの報告ディレクトリ内の証跡であり、実装hashの対象外である。

- `/dev/tty` を UTF-8 の `r` と `w` で別々に開き、両方の `isatty()` を確認する。
- 承認文を書き出して `flush()` してから、読取りハンドルで 1 行を読む。
- `.strip()` で入力を緩和せず、plan identity を含む承認文と末尾の改行まで完全一致させる。
- TTY が利用できない場合、TTY ではない場合、I/O エラー、EOF、Ctrl-C、文面不一致を固定の診断コードで区別する。いずれも承認を作らず停止する。

| 状態 | 固定診断コード |
| --- | --- |
| TTY open 失敗 | `supplemental_tty_unavailable_no_activation` |
| 読取りまたは書込みハンドルが非 TTY | `supplemental_real_tty_required_no_activation` |
| TTY の読書き・文字コードエラー | `supplemental_tty_io_error_no_activation` |
| EOF | `supplemental_approval_eof_no_activation` |
| Ctrl-C | `supplemental_approval_interrupted_no_activation` |
| 承認文の不一致 | `supplemental_exact_approval_not_given_no_activation` |

元の生成条件、budget、schedule、provider transport、HTTP retry 方針は変更しない。`--preview` を既定とし、`--execute` と新しい exact TTY approval が揃うまで補完送信を許可しない。

## 保持する実験と予約

元の canonical plan は `/root/minecraft-runs/goal006-remaining800-20261014T101309Z-v2/plan.json`。元の試行 **781 件 = 成功 761 + usage 既知の失敗 19 + HTTP 401 の失敗 1** と、未実行 **19 件** を区別して保持する。元の halt は `unknown_usage:auth_failure` のまま。

HTTP 401 は `buildings_100_v4_building_085/O` の OpenAI 生成、global index 940、attempt ID `2a912564cd574eeea2f6fa75d1683420`。元の ledger・approval・authority・予約 claim は変更しない。根拠は Goal006B の [個別失敗記録](../goal006b/auth_failure.json) と [会計監査](../goal006b/accounting.json)。

| 数値 | USD | 意味 |
| --- | ---: | --- |
| usage 既知分の保守的価格計算 | 20.857851 | 請求書照合をしていない価格計算額 |
| 元 HTTP 401 の usage 不明予約 | 0.165536 | 予約を保持する。実測 usage・請求額ではない |
| 現在の予算計上合計 | 21.023387 | 元の不明予約を含む |
| 補完 20 件の最大追加予約 | 5.293760 | 固定 payload と output 上限から算出 |
| 全 20 件を最大予約した場合の累積 | 26.317147 | USD 30 上限以内。請求額や消費予測ではない |

実請求額は UNKNOWN。補完が成功しても、過去の 401 の不明予約を自動で解放しない。

既存 781 件の GitHub Draft Release **414157245** と全 6 assets を保持する。archive asset **644830729** の SHA-256 は `c7897e5f121df0c08cc2a2a7e993f02fae3ac15fa5ca6f248e1e16cd14ca36a5`。B1 の [initial_integrity.json](initial_integrity.json) では現在の Release ID の metadata を GET し、全 6 assets の local SHA/size と remote digest を照合した。asset の再ダウンロードと mutation は 0。全 6 assets の取得・安全復元は保持した [Goal006B の検証証跡](../goal006b/existing781_github_verification.json) を参照し、その過去の検証時刻を B1 の新しい再取得として扱わない。同じ B1 監査で原本 inventory 3,740 files の hash 一致と、生成 entrypoint に一致するプロセスがないことも確認した。

補完 scope は最大 **20 送信**。global index 940 は元 401 からの明示的な手動認証再試行 1 件、941–959 は未実行 19 件の初回送信。OpenAI 10 件・Anthropic 10 件、元 payload・順序・model・prompt・schema・出力上限 32,768 を維持する。累積送信は既存 971 + 最大 20 = **991**、履歴 retry 4 + 手動認証再試行最大 1、自動 retry 0、未解決予約を含む累積 USD 30。既知失敗 19 件を再生成する範囲は含まない。

## 旧 plan の保管と新しい未承認 plan

旧 Goal006B plan `/root/minecraft-runs/goal006b-supplement20-20261015T195511Z/plan.json` は保持する。

- 旧 plan SHA-256: `87760792f4b63f887c9f1a2547dd8c8513356a33018978309cb63e4fc0487aa8`
- 旧 plan identity: `a1f167a8246d4f9f5a4c32a828b27156e2c2aa196c996f332e50de3617b8a4be`
- 旧 plan の準備証跡: [preparation_verification.json](../goal006b/preparation_verification.json)

TTY 修正で補完実装の hash が変わるため、旧 plan は新しい実装と互換にならない。旧 plan の bytes を保持したまま、`supplement_implementation_hash_mismatch` で拒否されることを確認した。旧 plan を書換え・再 hash せず、**修正コードを commit/push してから**新しい canonical plan を作成した。新 plan は `UNAPPROVED_SUPPLEMENT20`、`api_allowed: false`、未承認、補完送信 0。元原本の状態と既存 GitHub checkpoint に結び付いている。

| 最終証跡 | 状態 |
| --- | --- |
| 作業開始 HEAD | `3bbe3c64fe0080b43aa129dc05c909b6b797b1db` |
| 保持する main | `7309327efd1c19dda793b19e8d8e6ce0b5868db9` |
| 修正コードの commit / remote 一致 | `28fa9ea2f9bae114cea2be71f352ea6c955a5eaa` / 一致確認済み |
| 専用 branch | `research/server-only-20261013` |
| 新 canonical plan | `/root/minecraft-runs/goal006b-ttyfix20-20261015T201419244530Z/plan.json` |
| 新 plan SHA-256 | `088e7f5253c078ea310a7130e88c23df602a26910a61a0785266fd8af3c4ed21` |
| 新 plan identity | `206dfda6468a90b8d67700e3799571ff0de5aabe1263c15754efe242887f8922` |
| 原本・旧 plan 不変、新 plan 未承認 | [preparation_verification.json](preparation_verification.json)、[preview.json](preview.json) |

[final_integrity.json](final_integrity.json) と [独立した最終受入監査](acceptance_review.json) で、作成後の原本・旧報告・新planを再照合した。[github_save.json](github_save.json) は新plan作成前のコードcommitのremote一致を記録する。この報告だけを含む後続commitのSHAとremote照合結果は最終応答に記す。

2026-10-15 20:14:59 UTC に実 canonical plan の prepare と preview を完了した。新 plan が束縛する 44 code hashes は push 済み commit と一致した。旧 plan との差は `cli.py` の実装 hash と commit・作成時刻・plan/enabled directory のみで、その他の identity fields はすべて一致した。20 payload を元の schedule と照合し、global index 940–959 の順序、元 401 attempt ID、budget、model、prompt、schema が同じであることを確認した。

原本 3,740 files と元の凍結実装 39 files は hash 不変。新 directory にあるファイルは `plan.json` だけで、enabled directory・approval・ledger・authority・lock・新 reservation は作成していない。元の halt と USD 0.165536 の未解決予約も維持している。prepare は 30.835 秒、preview は 6.662 秒、最大 RSS は 412,068 KiB。Python network を遮断し、provider キーを値を読まずに環境から除いて検証した。

## 検証と今後の操作

今回の [tests_summary.json](tests_summary.json) は **77 unique cases の最終結果が PASS、未解決 failure/error/skip は各 0**。内訳は planning/runner/checkpoint の CPU 回帰 45、TTY/CLI 32（うち実 TTY 26）。[core_tests.junit.xml](core_tests.junit.xml)、[tty_cli_tests.json](tty_cli_tests.json)、[tty_cli_tests.junit.xml](tty_cli_tests.junit.xml)、[tty_double_launch.junit.xml](tty_double_launch.junit.xml) に証跡を保存した。

実 PTY の成功系、stdin が `/dev/null` でも controlling TTY から承認できること、EOF・Ctrl-C・切断・非 TTY・open/IO error・不正 UTF-8・前後空白を含む不一致入力の停止、pipe/env からの承認拒否を確認した。二重起動の fixture は初回、同時 status 確認の一方が nonblocking source lock で正しく停止する可能性を考慮せず失敗した。最初の実 prompt を確認してから二つ目を起動する手順へ fixture を修正し、その 1 case を再実行して PASS。最初の失敗 XML も保持する。production code の追加変更はなく、最初の処理が fake response 20 件を保存した後、二つ目の exact consent は既存 authority により拒否され、重複送信 0 を確認した。

実 API の代わりに fixture を使い、元の approval や authority を作り直していない。テストの呼出しは再実行を含め 78 回で、production `--execute` の起動は 0。Goal006B の過去の 51 unique cases は [validation_summary.json](../goal006b/validation_summary.json) に保持し、今回の実行結果へ加算していない。

確認用の絶対パスコマンドは次のとおり。`--preview` は approval・authority・ledger の作成や API 送信を行わない。

```bash
/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python -B /root/workspace/Minecraft-view-dataset/offline_evaluation/operations/goal006b/cli.py --plan /root/minecraft-runs/goal006b-ttyfix20-20261015T201419244530Z/plan.json --preview
```

将来、上記の **未実行 19 + 認証再試行 1、累積 991 calls・USD 30、自動 retry 0** を明示承認する場合のコマンド。今回この有料実行は行わない。

```bash
/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python -B /root/workspace/Minecraft-view-dataset/offline_evaluation/operations/goal006b/cli.py --plan /root/minecraft-runs/goal006b-ttyfix20-20261015T201419244530Z/plan.json --execute
```

実行時には当日 UTC の公式料金証跡と両 provider キーの存在が必要。料金証跡の期限切れは補完側だけへの追記で更新し、元 plan の pricing revision を変更しない。承認文は新 plan identity を含む全文を実 `/dev/tty` から入力する。キー更新の申告・CPU テスト成功・preview だけでは有料生成の承認にならない。

補完の保存は既存の `LOCAL_VERIFIED` / `github_status: NOT_UPLOADED` の動作を維持する。元の 781 件 GitHub checkpoint が検証済みであることと、将来の補完結果の GitHub 保存状態は分けて報告する。
