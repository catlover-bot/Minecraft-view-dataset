# Goal006 残800入力の操作手順

このGoalでは研究APIの追加送信を行っていない。下記の計画は、完了済み第1バッチ160入力と同じ固定実験の残50建築・800入力だけを扱う。準備完了は、全60建築・960入力の実行完了を意味しない。

Canonical plan（作成・検査後の実identity/SHAは準備報告で確認）：

`/root/minecraft-runs/goal006-remaining800-20261014T101309Z-v2/plan.json`

初回起動は利用者が両providerキーを設定した本物の端末で行う。実行範囲は新800送信・retry0、旧消費を含め累積990送信・retry4・USD30、output32768、concurrency1である。旧消費は190送信・1,828,481 tokens・保守的費用計算USD4.048922を一度だけ引き継ぐ。USD30は追加30ドルでも完了予算の保証でもない。サーバー料金は別である。

## 開始・同台帳での再開

推奨する開始コマンドは次の1本。初回はplan identityの先頭12文字を含む完全一致の確認文字列を実TTYで入力する。Codexは入力しない。既に承認された同じplanの再開では、既存approvalと同じledgerを再利用する。

```bash
/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python -B /root/workspace/Minecraft-view-dataset/offline_evaluation/continuation_v1/launch.py --plan /root/minecraft-runs/goal006-remaining800-20261014T101309Z-v2/plan.json --github-checkpoints
```

確認文字列の形式は `APPROVE GOAL006 <plan identityの先頭12文字> 800 SENDS 0 RETRIES USD30 TOTAL`。実際の値は作成後の準備報告およびランチャー表示を使用する。この提案への承認は保存されていない。

同じコマンドの末尾に `--preview` を付けると、キーや承認入力なしで表示を確認できる。previewはenabled・authority・reservationを作らない。`--github-checkpoints` は将来の区間保存をGitHub Draft Releaseまで行う指定であり、preview中にuploadは行わない。

50建築の元順序を保った10建築・160入力ごとの5区間を一つの承認で続ける。区間境界はlocal index 160/320/480/640/800、全体schedule indexは160..959である。区間ごとの予算リセットはない。provider/input_idの同じ組を二重送信しない。成功・生成失敗・送信不明を再送しない。

## 読取り状態確認・CPU評価

リポジトリ内で実行する。statusと過去結果の評価には当日の料金確認を要求しない。

```bash
cd /root/workspace/Minecraft-view-dataset
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.continuation_v1.cli status --plan /root/minecraft-runs/goal006-remaining800-20261014T101309Z-v2/plan.json
```

```bash
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.continuation_v1.cli evaluate --plan /root/minecraft-runs/goal006-remaining800-20261014T101309Z-v2/plan.json
```

未承認のevaluateはenabledを作らず停止する。承認後は新しく完了した継続出力だけをCPU評価し、既存の評価行をplan・attempt・GTのhashで確認して再利用する。statusは未試行をNOT_RUNとして保持する。通信不明や未精算予約のhaltを削除したり、ledgerを再初期化したりしない。

## UTC日付変更時

日付をまたいだ場合、次の送信の予約前に停止し、進捗・未試行数・同じ再開コマンドを表示する。既存approvalとledgerは保持する。新しい公式確認JSONを作る前に、固定モデルについて公式文書を実際に取得し、取得UTC・URL・料金・context/output上限・native structured-output対応・提供状況を記録する。モデル照会API、token-count API、生成APIによる確認は不要である。日付だけを書き換えない。

以下の `NEW_RECORD` は、実際の取得証跡を含む新JSONの絶対パスを利用者が設定するための変数である。

```bash
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.continuation_v1.cli refresh-pricing --plan /root/minecraft-runs/goal006-remaining800-20261014T101309Z-v2/plan.json --pricing "$NEW_RECORD"
```

正規CLIはhash付き `pricing-revisions/0001.json` 以降へ追記し、元plan・approval・ledgerを変更しない。料金、context/output上限、モデルID許可、課金条件が変われば停止する。自動fallback、cap拡張、日付偽装、検証の無効化は行わない。確認後は上の同じランチャーで未試行分だけを再開する。

## 区間保存・保存失敗の再試行

各応答は先にディスクへ永続化される。160入力の区間完了時と安全な停止時に、active requestがない状態で集計とpoint-in-time checkpointを作成する。旧source・旧190応答は検証済みasset ID/SHA参照を再利用し、区間ごとに重複uploadしない。`--github-checkpoints` 指定時は新規Draftへ保存し、再取得SHAと安全な展開を確認する。

保存に失敗しても生成結果・usage・予約はディスクに保持し、生成失敗へ書き換えない。境界で安全に停止する。表示された実際の保存済みcheckpointパスを `SAVED_CHECKPOINT` に指定し、保存だけを再試行する。

```bash
/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python -B /root/workspace/Minecraft-view-dataset/offline_evaluation/continuation_v1/manage.py checkpoint-upload --checkpoint "$SAVED_CHECKPOINT"
```

このコマンドは生成処理や承認を呼ばない。既存assetを上書きせず、同じpackageの保存を再確認・再開する。別途、新しいpoint-in-time保存を明示的に作成する場合は、一意で未作成の保存先を `NEW_CHECKPOINT` に指定する。

```bash
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.continuation_v1.cli checkpoint --plan /root/minecraft-runs/goal006-remaining800-20261014T101309Z-v2/plan.json --destination "$NEW_CHECKPOINT" --github
```

コピーは復元・検証用であり、有料再有効化には使用できない。実行はcanonical planと同じledgerだけで継続する。

## 全960の集計入口

一意で未作成の出力先を `NEW_AGGREGATE` に指定する。

```bash
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.continuation_v1.cli aggregate --plan /root/minecraft-runs/goal006-remaining800-20261014T101309Z-v2/plan.json --destination "$NEW_AGGREGATE"
```

parent160と継続800を元manifestへ照合し、semantic protocolと各runのprotocolを区別する。未試行を生成失敗0に変換しない。全60建築の結果が揃うまでは記述統計だけであり、現在の途中結果から対象や分析familyを変更しない。完了後は凍結済みanalysis_v4のbuilding単位paired集計・bootstrap・sign-flip・Holmの3 families（8/4/8比較）を使う。このGoalの最終推論検証にはfixtureだけを使用し、未実行800結果を作らない。

## 端末と終了状態

tmuxは必須ではない。既存tmuxを使う場合、親SSH端末でexportしたキーが既存paneへ自動継承されるとは仮定しない。実際に起動するpane内でキーを非表示入力して環境へ設定し、キーをファイルやコマンド引数へ保存しない。確認文字列はそのpaneの実TTYで利用者が入力する。OS変更、SSH変更、ユーザー切替、tmux導入、キー再登録はこのGoalに含まれない。

ランチャー終了コードは、0=範囲完了またはpreview、2=未承認、3=キー不足、4=実行状態の安全停止、5=検証・入出力エラー、6=料金鮮度待ちやcheckpoint保存失敗など運用停止。理由と現状を確認してから同じplanへ戻る。不明usage・未精算予約・identity不一致などを消して再開しない。
