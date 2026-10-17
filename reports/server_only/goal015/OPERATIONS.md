このrunは未承認の準備状態で、研究API送信0。以下の開始コマンドは**利用者が完了報告を確認した後、別実行で実TTYへ承認を入力するため**のもので、Codexは実planへ入力していない。

作業ディレクトリは `/root/workspace/Minecraft-view-dataset`。既存 `.venv-server-v2-20261013/bin/python` を読み取り利用する。新install不要。すべて `-B` で旧領域へpycacheを書かない。canonical planは `/root/minecraft-runs/goal015-20261017T170546Z/plan.json`。Git内のplan copyはレビュー用で、生成起動はできない。

キー不要・書込みなしの確認：

```bash
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.experiments.goal015.cli preview --plan /root/minecraft-runs/goal015-20261017T170546Z/plan.json
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.experiments.goal015.cli status --plan /root/minecraft-runs/goal015-20261017T170546Z/plan.json
```

preview/statusはapproval/authority/ledger/enabled/lockを作らない。公開済みprepared registry、実plan、code/spec/input hashを照合する。将来の実行では表示されるplan ID・80 calls・S/F・USD5・retry0の完全一致承認が必要。/dev/ttyの別read/write handleとflushを使い、Enter/不一致/EOF/nonTTY/read failureは未承認で終了する。

利用者による将来の開始コマンド：

```bash
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.experiments.goal015.cli start --plan /root/minecraft-runs/goal015-20261017T170546Z/plan.json
```

実TTYを持つ利用者のterminalで実行する。実planへの承認文を自動pipeしない。準備完了表示だけでは支払い承認にならない。初回起動はcommitted registryも照合する。価格資料が24時間を超えた場合は承認前に停止する。指定モデルの停止日以降も停止し、自動fallbackしない。USD5は80件完走を保証せず、次paired blockの予約が不足すれば途中停止する。

同じidentity・authority・ledgerからNOT_SENTだけを継続する：

```bash
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.experiments.goal015.cli resume --plan /root/minecraft-runs/goal015-20261017T170546Z/plan.json
```

送信済み／不明送信は再送しない。unknown usageのholdを維持する。契約不一致、別root/copy、改変payload/code/specを通すために新planへ複製しない。必要な例外や契約変更は別の明示承認を要する。401/403/520/timeoutはmodel-output failureと区別してstatusとrawを確認する。秘密をterminalログやGitへ載せない。

24時間を超えた資料は、同planの独立evidenceとして更新できる。次のcaptureは公式ドキュメントGETだけを行い、model/認証/token-count APIは呼ばない。新規出力directoryを指定する。

```bash
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.experiments.goal015.cli capture-pricing --plan /root/minecraft-runs/goal015-20261017T170546Z/plan.json --output-dir /root/minecraft-runs/goal015-20261017T170546Z/pricing-refresh-01
```

captureの状態はREQUIRES_HUMAN_CONTRACT_REVIEW。取得資料で両snapshotの存続、endpoint/schema/sampling/capとstandard価格を確認する。元 `pricing_contract.json` と**同じ契約であることを確認できた場合だけ**、具体的review-noteを記録してsealする。変更があればこのsealは使わず新承認へ進む。

```bash
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.experiments.goal015.cli seal-pricing --plan /root/minecraft-runs/goal015-20261017T170546Z/plan.json --capture /root/minecraft-runs/goal015-20261017T170546Z/pricing-refresh-01/capture.json --reviewed-contract /root/minecraft-runs/goal015-20261017T170546Z/pricing_contract.json --review-note '取得した全公式資料で価格・snapshot・対応条件を照合し、元の契約との一致を確認した。' --output /root/minecraft-runs/goal015-20261017T170546Z/pricing-refresh-01/reviewed.json
```

その後startまたはresumeへ `--pricing-evidence /root/minecraft-runs/goal015-20261017T170546Z/pricing-refresh-01/reviewed.json` を付ける。新contractを自動承認しない。timestampを書き換えただけの資料、欠けたURL集合、古い本文、改変hashは拒否する。価格証拠追記は呼出し回数・台帳をリセットしない。

保存応答だけのCPU評価とpaired集計：

```bash
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.experiments.goal015.cli evaluate --plan /root/minecraft-runs/goal015-20261017T170546Z/plan.json
```

新しい `evaluations/<UTC>/` に全80行のevaluation_80.jsonとsummary.json、適格応答のparsed/voxel/metricsをexclusive保存する。送信がなければ80行NOT_SENT/NULL、GT読取0。provider/生成adapterをimportせず研究API0。既存execution lockがあれば読み取りopenで取得し、生成と同時評価しない。二度目も前の評価を上書きしない。

既存evaluationの集計だけを再作成する場合は、直前コマンドが返した実ファイルへ指定する：

```bash
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.experiments.goal015.cli aggregate --plan /root/minecraft-runs/goal015-20261017T170546Z/plan.json --evaluation /root/minecraft-runs/goal015-20261017T170546Z/evaluations/preparation-check/evaluation_80.json
```

上のpreparation-checkは今回作成した未送信80行の確認用artifactで、実結果ではない。採取後は新しいevaluation pathを使う。再集計は生成やGT再評価をしない。

再集計には、evaluateが作成した未変更のevaluation_80.jsonだけを使う。aggregateはplan IDと予定行の対応を確認するが、保存された数値行を信頼して集計し、raw/parsed/voxel/metricsのhashを独立に再照合しない。statusは元のsummaryからevaluationファイルへのhash束縛を確認する。由来や変更の有無が不確かなら、保存rawからevaluateを再実行する。aggregateは上記の標準出力コマンドを使い、evaluations配下への `--output-dir` は指定しない。この任意出力のsummaryにはevaluationへの束縛がなく、後続statusの検証に通らない。

生成中は各建築終了時／安全停止時にlocal checkpointを作る。評価後の保存やlocal保存失敗時の保存専用retry：

```bash
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.experiments.goal015.cli checkpoint-local --plan /root/minecraft-runs/goal015-20261017T170546Z/plan.json
```

実行台帳がない準備状態ではcheckpointを生成しない。実行後はplan、enabled台帳/応答と既存evaluationsを保存する。原description/input本文のcopyは含めない。canonical原本は保持し、復元copyから生成は不可。archiveはmanifest/全member SHA、path traversal/link/duplicateと展開size上限を検証してLOCAL_VERIFIEDを記録する。

将来の新Draftへの保存は、上のコマンドが返す実checkpoint directoryと新しい検証先を引数へ渡す。CLIは次の形式で存在する（`CHECKPOINT_DIRECTORY` と `NEW_VERIFICATION_DIRECTORY` は実値へ置換）：

```bash
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.experiments.goal015.cli checkpoint-upload --checkpoint-dir CHECKPOINT_DIRECTORY --verification-dir NEW_VERIFICATION_DIRECTORY --allow-create
```

一覧で一致する新Goal015 Draftを探しnumeric Release IDで確認し、asset IDでdownloadしてSHA/安全展開を検証する。既存assetは上書きしない。一度得たRelease IDは同コマンドの `--release-id ID` で再利用できる。失敗時はlocalを残して新しいverification directoryへ保存のみ再試行する。生成を再実行しない。GITHUB_VERIFIEDは実際の再取得検証後だけ表示する。今回のGitHub Draft通信はfake fixtureだけで、実Draft作成／uploadは0。

テストは新Goal015だけを既存venvで行った。実PTYは隔離fixture planに限定。mock80の完走と欠測→resumeの検証は研究API呼出しに数えない。完成時のTEST_SUMMARY、MOCK_END_TO_END_RECEIPT、INTEGRITY_AFTERとEXECUTION_BINDINGで範囲を確認できる。
