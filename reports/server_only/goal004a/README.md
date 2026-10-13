# Goal004A — 承認TTY入出力の修正

状態は **TTY_FIXED_READY_AWAITING_EXPLICIT_APPROVAL**。今回の研究API追加送信0回。実planの確認文字列は入力しておらず、新旧ともenabled・authorityは未作成。親campaignの14送信・retry4・173509tokens・保守的USD0.25857425を維持した。

利用者の2026-10-13 18:30:45 UTCのログだけで入力ミスとは判断していない。対象サーバーのPython **3.12.3**、制御端末のある `pty.fork()` 子プロセスで、旧 `open('/dev/tty', 'r+', encoding='utf-8')` は **`io.UnsupportedOperation: File or stream is not seekable.`** を発生させた。errnoはnull、OSErrorのサブクラスである。旧コードではこの例外をFalseへ丸めるため、確認文表示前でもNOT_APPROVEDになり得た。`r` と `w(buffering=1)` の別ハンドルは同じ実PTYで成功した。[最小再現スクリプト](tty_reproduction.py)、[実測結果](tty_reproduction.json)。

実行コードの変更は `offline_evaluation/resmoke_v2/cli.py` だけ。読取り・書込みを別ハンドルにし、確認文を表示してflush後、/dev/ttyから1行だけ読む。改行で確定した完全一致だけがTrueになる。空Enter・不一致・EOF・一致文字列途中のEOF・Ctrl-CはFalse。端末open失敗は `TTY_UNAVAILABLE`、read/write/flush/closeや文字デコード失敗は `TTY_IO_ERROR` として、型と安全なerrnoだけを報告する。例外本文や入力全文を診断へ出さない。stdin・環境変数・--yes・承認ファイル編集による迂回は追加していない。

コードcommitは `fd61319b87d019a527c3acd3660f481d4e7dd5e4`。開始HEAD `fb2eba5bd172d5d147129a34b357e83da0d7a10e` と既存の `research/server-only-20261013` を継承した。root、workspace-write / auto_reviewを維持。TTY問題をsandbox設定の不一致と扱わず、OS・SSH・ユーザー・モデル・研究条件は変更していない。

実PTY17ケースと既存ランチャー・承認・実行器の回帰を合わせ、**81 passed / failures0 / errors0 / skipped0**。pytest10.45秒、実時間10.75秒、最大RSS50068KiB。各子プロセスは5秒の待機期限と終了・回収処理を持つ。確認文の実表示と入力前の待機、fixture専用語の一致／不一致／空Enter／EOF／Ctrl-C、stdin pipe/null、制御端末なし、read/write/flush失敗、不正UTF-8を検証した。成功語を送った相手はfixture確認関数だけで、実planや実ランチャーの自動承認は行っていない。200GT往復や過去1200出力の全再評価は再実行していない。[テスト結果](tests.json)、[ログ](tests.log)、[JUnit](tests.junit.xml)。

元planは保持し、既存prepareで一意な後継disabled planを作成した。[機械的な新旧差分](plan_succession.json)、[新planの保存用コピー](disabled_plan.json)。旧planの実装hashをその場で書き換えていないため、旧planは修正後の実装では有効化できない。

| 項目 | 元plan | 後継plan |
|---|---|---|
| 実パス | `/root/minecraft-runs/goal004-resmoke-20261013T175539Z/plan.json` | `/root/minecraft-runs/goal004a-ttyfix-20261013T194600Z/plan.json` |
| ファイルSHA256 | `d13b406df1bdd5f2fee5ca5821691b4fcb9c06875da4d2120d24104f45d2b74a` | `c1fbd060b0d7459ad26d74b27994f9f2b891264a21d4835bf81bec32679c1030` |
| identity SHA256 | `047028e5645a90a94de70137d4b9ab579c9e56a08e583d723654276d2ac2cd77` | `b522f5bbc1d17526b4366cd27a8f1968379f007a73f38c2a471b840bb6a549f2` |

identity内の差分はcli.pyの実装hash、commit、plan_path、enabled_directoryだけ。envelopeの差分はidentity_hashとcreated_at_utc。060/075、全16入力本文・input ID・モデル、schema、prompt、provider順、output32768、追加16送信・retry0、累計30送信・retry4・USD5、token上限、価格、親ledger/state、authority pathは同一。後継planのpreviewは別cwd `/tmp` とsocket接続禁止でexit0を確認した。[previewログ](launcher_preview.log)。

実サーバーのUTC日は **2026-10-13**。保存済み公式記録の実取得日時は **2026-10-13T17:45:27Z** で、同じUTC日のため今回の再取得は不要と判断した。日付だけの書換え・時刻偽装・期限チェックの無効化は行っていない。[既存公式記録](../goal004/official_pricing.json)と、そのURL・確認内容を保持する。実アカウントでのモデル/schema受理はNOT_RUN。

利用者が両APIキーのあるサーバーターミナルで実行する開始コマンドは次の1本。cwdには依存しない。

```bash
/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python -B /root/workspace/Minecraft-view-dataset/offline_evaluation/resmoke_v2/launch.py --plan /root/minecraft-runs/goal004a-ttyfix-20261013T194600Z/plan.json
```

全対象・上限・hash・保存先を確認した後、端末へ入力する実際の確認文字列は次のとおり。**Codexはこの文字列を実planへ入力していない。**

```text
APPROVE GOAL004 b522f5bbc1d1 16 SENDS 0 RETRIES USD5 TOTAL
```

確認前にはenabled・authority・送信予約を作らない。承認後は固定16件だけを生成し、CPU評価・集計する。本評価960への自動進行や追加retryはない。残る運用上の制約はGoal004と同じで、公式確認日の翌日以降は送信前に停止し、未知usageや送信不明は予約を保持して自動再送しない。既に有効化した台帳を別planへ移して初期化する機能はない。

旧69証拠ファイル＋4制御ファイル、元plan、親ledger/stateの不変を確認した。[事前状態](preflight.json)、[最終照合](final_validation.json)。GitHub保存は専用ブランチと新規未公開Draftだけを使用し、小さい後継planとTTY関連証跡を保存する。保存・再取得の確定情報はcheckpoint索引に記録する。
