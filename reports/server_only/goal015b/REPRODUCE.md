Goal015Bは保存応答だけを読む診断である。新規取得・install・API・料金更新・親子launcher実行は不要。既存Pythonと依存をそのまま用いる。

1. UTCと固有suffixを持つ新しい`/root/minecraft-runs/goal015b-...`をexclusiveに作る。親と子の状態、HEAD/branch/status、親rb flockを確認し、`preservation.capture_snapshot(new_run)`で入力を固定する。実行中の生成lockがあれば停止する。
2. `python -B -m pytest -q -p no:cacheprovider offline_evaluation/analysis/goal015b/tests --basetemp=<new-run>/tests --junitxml=<new-run>/tests.xml`を実行する。初期fixtureの非空が通ることを先に確認する。過去の全suiteは対象外。
3. `python -B -m offline_evaluation.analysis.goal015b.audit --run-root <new-run> --report-root <new-run>/reports`を実行する。new-run自体は先に作る。`audit/`と表の既存出力は上書きしない。独立extractorは重複キー・非有限JSON・型不整合を拒否し、unknown/nontext/refusal/multipleを記録する。失敗prefixは観察だけで実行しない。
4. 旧O比較は`python -B -m offline_evaluation.analysis.goal015b.old_comparison --run-root <new-run> --parent-plan /root/minecraft-runs/goal015-20261017T170546Z/plan.json --report <new-run>/reports/INPUT_DELTA_AND_REPLAY.md`で作る。ID順固定選択、各Builder1件、前読取りhash記録を維持する。原GTを使わない明示self-reference replayである。
5. `preservation.finalize_snapshot(before_path, new_after_path)`と、phase4の`INPUT_SNAPSHOTS.jsonl`に記録した各sourceのhash再確認を行う。新しいrun内のreceiptと表を束縛して保存する。

全Pythonで`PYTHONDONTWRITEBYTECODE=1`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。推論モデルは使わず、CPU作業を順番に実行する。全体CPU900秒、開始からwall1800秒、RSS目標2GiB/上限4GiB、新run1GiBを維持する。audit CLI自体はCPU450秒/address-space4GiBを設定し、network audit hookでconnect/resolve/requestを拒否する。OSによる完全隔離の主張ではない。個別CLI上限だけで全Goalの累積上限を保証しないため、先行stageの実測も合算し、超過見込みなら残件を記録して停止する。

本実行ではOpenAIの返却formatに`description:null`が追加されていることを検出した。最初の診断ではformat全体を比較していたが、最終表ではformat全体一致とschema/契約一致を分離した。最初の診断出力は新run内に残し、最終実行も新しい出力先へ保存した。この修正はGoal015Bの診断表の説明に関するもので、親の生成・評価・ledgerを変更していない。

NULLは欠測・未実行を意味する。reasoning usageがないAnthropicを「reasoning0」と解釈しない。集計の`reasoning_tokens_raw_present_sum=0, reasoning_usage_rows=0`は観測値がないことを示す。HTTP statusはledger根拠であり、raw JSONにHTTP headerが保存されているとの主張ではない。
