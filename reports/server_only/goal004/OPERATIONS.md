# Goal004 利用者用ランチャー

研究APIの追加送信は0回。実アカウントでのnative structured output受理、新Eの成功、Pの空出力解消はいずれもNOT_RUN。現在はdisabled計画で、enabled run・新承認・新台帳は存在しない。

キーを設定済みの同じサーバーターミナルで、次の一つのコマンドを実行する。作業ディレクトリに依存しない。`OPENAI_API_KEY` と `ANTHROPIC_API_KEY` は親シェルから読み、値は表示・保存しない。

```bash
/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python -B /root/workspace/Minecraft-view-dataset/offline_evaluation/resmoke_v2/launch.py --plan /root/minecraft-runs/goal004-resmoke-20261013T175539Z/plan.json
```

ランチャーは全16対象、provider順、固定モデル、input/schema/prompt/実装hash、出力上限32768、追加送信16・retry0、既存費用、累積USD5、保存先とキー存在boolを表示する。表示を確認した利用者が `/dev/tty` に完全一致の確認文字列を入力した場合に限り、新enabled runを作り、生成・CPU評価・集計を順に実行する。Enter・EOF・不一致は送信0回で終了し、stdin・環境変数・承認省略フラグでは承認できない。Codexはこの確認を入力していない。

`--preview` を同じコマンド末尾へ付けると読取りのみ。今回この実在ランチャーを `/tmp` から、socket接続禁止の状態で実行しexit0を確認した。[実測ログ](launcher_preview.log)。

承認範囲はbuilding060/075 × O/J/P/新E × OpenAI/Anthropicの固定16入力。要求モデルは `gpt-5-mini` と `claude-haiku-4-5-20251001` のまま。順序は各建築のO→J→P→E、それぞれOpenAI→Anthropic。すべて初回最大1回で、追加retry・モデル置換・建築の差替え・E切詰め・本評価960への自動進行はない。

| 上限・予約 | 値 |
|---|---:|
| 既存送信 / retry / tokens | 14 / 4 / 173509 |
| 既存保守的USD | 0.25857425 |
| 追加送信 / retry / concurrency | 16 / 0 / 1 |
| 累積送信 / retry | 30 / 4 |
| 共通出力上限 | 32768 tokens |
| 追加token予約上限 / 累積上限 | 5324288 / 5497797 |
| 全16件の最大予約額 | USD 4.235008 |
| 旧消費＋全16件の最大予約額 | USD 4.49358225 |
| 累積USD停止上限 | 5.00 |
| 追加の理論上限 | USD 4.74142575 |

予約は各モデルの全contextを入力として確保し、schemaによる追加system promptもその範囲に含め、出力上限をさらに加える。理論的な予約額は実費予測・請求保証・全16件完了保証ではない。HTTPエラーやtimeoutでusage不明なら予約を保持して停止する。

出力上限32768は新しい未承認の提案である。旧16384からの変更、旧累積20送信から30送信への変更を、今回の表示と明示確認の対象に含める。旧 `manual_confirmed=true` は流用しない。

同じコマンドの再実行は、承認済みなら元のcanonical ledgerだけを再開する。成功・失敗・送信不明の入力を再送せず、予算も初期化しない。固定親campaignに対するauthority記録が別計画・別ディレクトリの再有効化を拒否する。生成中は旧campaignの4既存lockを内容不変のまま保持し、親ledger/state SHA・使用量が変わっていれば停止する。

公式確認日は2026-10-13 UTC。初回開始・再開とも同日の公式確認を要求する。日付が変わると送信前に停止するため、必要になった場合は公式仕様・料金の再確認を行う。既に有効化されたrunでは同じ台帳を保持する更新が必要で、別run作成でこの制約を回避しない。新確認をせず自動で有効期間を延ばす機能はない。固定モデルのdeprecated/legacy状態と廃止予告は[公式確認](official_support.md)を参照。

初期有効化がauthority記録後に中断した場合も安全側に停止する。部分ファイルの自動削除・新予算によるやり直しはせず、記録を検査する。600秒のHTTP timeoutは長出力の完了保証ではない。観測が全16件揃ったexit0と、非空再建築の成功率・GT品質は別々に報告する。

CPU結果・台帳・rawは新しい `.../goal004-resmoke-20261013T175539Z/enabled/` 内へ保存される。旧smokeと結果を混合せず、smokeで有意差検定を行わない。保存用のplanコピーはcanonical path検査によりその場で有効化できない。
