Goal014は完了した。対応画像対照には識別信号が見られたが、同じsummary CLIPScoreと保存Oのsurface F1@1との係数は両Builderとも0付近だった。未切捨て群でもoccupancy IoUとの正の関連には変わらなかった。これはGoal013Aの結果既知の **POST_HOC_EXPLORATORY** であり、元score・元CI・原RQ2は変更しない。

| 問い | 主な結果 | 実分母 |
| --- | --- | --- |
| Q-A：対応画像対照 | 建築等重みD=0.03372、95% CI [0.02802, 0.03944]。W=0.82321 [0.77532, 0.86436] | 178 summary、60 query建築、固定60候補gallery |
| Q-B：surfaceとの関連 | rho_surfaceはOpenAI0.02015、Anthropic0.02519。paired delta_rhoは0.09707 [−0.03352, 0.23132]／0.14908 [0.02518, 0.28323] | 共通valid175／177行、各60建築。元IoU集合と同一 |
| Q-C：未切捨て感度 | rho_IoUはOpenAI−0.16671 [−0.33050, −0.00789]／Anthropic−0.21880 [−0.39854, −0.03583] | 118／119行、各59建築 |

全12項目と切捨て群を含む4感度表は [RESEARCH_DECISION_JA.md](RESEARCH_DECISION_JA.md)、[ESTIMATES_12.csv](ESTIMATES_12.csv)。全row ID/mask、各実分母、token長・provider構成、finite/undefined数は [RESULTS_SUMMARY.json](RESULTS_SUMMARY.json) に残した。12項目すべて2000/2000 finite、固定CI gate PASS、planned60建築の同じPCG64 seed20261017 drawを共有した。新p値・significance label・Holm family・equivalence検定は0。

Q-AのWは59他候補との比較で完全tieを半点とする割合で、top-1精度ではない。生成元以外の建築は来歴上非対応という対照であり、内容が確実に偽になる負例ではない。D/Wはsummaryを建築内平均した後、建築を等重みにする。CIは固定galleryに条件付けたquery建築の不確実性で、新しいgallery/cohortへ一般化しない。

Q-BのAnthropicではsurface側の係数がIoU側より高い方向のpaired差がある一方、surface自体の係数は0付近である。この2つを混同しない。surface F1@1はGoal010v2で保存済みの6近傍boundary voxel中心距離指標で、空洞内壁を含む。元occupancy最適shiftをそのまま継承し、新geometry計算・位置合わせは0。画像の可視外観・mesh・意味的間取りを測ったとはしない。Q-B/Cのpooled点推定は記述行等重みで、建築等重みではない。

Q-Cの未切捨て119／切捨て59という元flagを変更せず、Builder validとの共通集合を数えた。切捨て群はOpenAI57行44建築、Anthropic58行46建築。provider・文長・内容・難度も変わる選択なので、群差は切捨ての因果効果ではない。未切捨て群をprimaryへ昇格していない。

**固定と実行。** 新plan・config・コードは計算前commit `6c7019094b0afdfe28ee88d5cc086797c62e457a` に固定した。[ANALYSIS_PLAN.md](ANALYSIS_PLAN.md)、[CONFIG.json](CONFIG.json)、[EXECUTION_START.json](EXECUTION_START.json) に原本・code・score・configの対応を保存した。原本runは `/root/minecraft-runs/goal014-20261017T145709Z`。既存の `/root/minecraft-tools/goal013a-20261017T131846Z/venv` を読取り利用した。

モデルは同じ公式OpenAI CLIP ViT-B/32、CPU float32、prefix `A photo depicts `、truncate=True、224 resize/center crop、単画像q=2.5×max(cosine,0)を計算してから6値平均。score IDは `MEAN6_CLIPSCORE_VITB32_CPU_FP32_SUMMARY_ONLY_GOAL013A` のまま。重みSHA256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`、tokenizer・変換・依存は元configと照合した。元画像・原文の加工や補完、別12視点、別モデル、prefix/crop変更は0。

Goal013A原本runに永続特徴artifactはなかった。[FEATURE_AVAILABILITY.json](FEATURE_AVAILABILITY.json) を確認後、**360画像＋178textを一巡だけ再encode**した。著者実装parityの固定1対では画像4／text4のforward入力を別計数した。再推論0回とはしていない。外部APIは使用せず、主行列の64,080 feature対は内積等のローカル算術である。[INFERENCE_COUNTS.json](INFERENCE_COUNTS.json)。

実著者関数parityはPASS。token/tensor/normalized feature/per-image qが一致し、独立scalarとの差は2.84×10⁻⁸。元1068 viewのreplay最大差1.79×10⁻⁷、178 mean6最大差1.39×10⁻⁷で、固定atol=rtol=1e-5を満たした。元scoreは置換しない。[REFERENCE_PARITY.json](REFERENCE_PARITY.json)、[SCORE_REPLAY.json](SCORE_REPLAY.json)。実token hash178件は一致し、Gemini054/061は欠測のまま。[TOKEN_REPLAY.json](TOKEN_REPLAY.json)、[QA_ROWS_180.csv](QA_ROWS_180.csv)。

特徴はprivateのfloat32数値npy、行列は178×60の数値npyとしてallow_pickle=Falseで保存し、再読込時にhash・順序・shape・dtypeを確認した。180予定行のうち空summary2行はNULLで保持した。特徴metadataは元summary/token/画像/score/model/configのhashへ束縛し、元原文・token列を複製していない。行列・features・全bootstrap drawはprivateに残し、Gitへ送っていない。[QA_FREEZE.json](QA_FREEZE.json) の原本hash参照を利用する。Q-A凍結後にのみBuilder/surface outcomeを結合した。

**検証と保全。** 新規fixture81件PASS、最後の入口修正に関係する5件も再確認してPASS。旧全実験・旧全テストの再実行は0。source/feature順序、元score差、clip後mean6、非有限・zero norm・空summary、候補自身除外、並替え、exact tie、bootstrap多重度/定数/欠測frame/paired差、Q-C flag/分母、旧score/statisticsへの書込み拒否を確認した。[FIXTURES.junit.xml](FIXTURES.junit.xml)、[PIPELINE_FINAL.junit.xml](PIPELINE_FINAL.junit.xml)。

独立検算は [INDEPENDENT_QA_REVIEW.json](INDEPENDENT_QA_REVIEW.json)、[INDEPENDENT_STATISTICS_REVIEW.json](INDEPENDENT_STATISTICS_REVIEW.json)、[INDEPENDENT_BINDING_REVIEW.json](INDEPENDENT_BINDING_REVIEW.json)。全12点推定・12CI・全paired差、固定抽出行列、選択した48 draw推定、1068 matched scalarと固定54 nonmatching scalar、360 O outcome identityと実maskを照合した。追加モデル推論と全bootstrap再実行は0。

開始/終了各1回の全体保全はPASS。旧28,391保護files、旧tracked18,335、27保護tree roots（既存venv/cache、Goal013A専用環境/runを含む）のbytes・membership・symlinkが一致した。14 read-only locksとPython監査guardを使用し、旧path書込み・network attemptは0。OS隔離保証ではない。[INTEGRITY_BEFORE.json](INTEGRITY_BEFORE.json)、[INTEGRITY_AFTER.json](INTEGRITY_AFTER.json)。

計算プロセス全体CPU104.91秒／elapsed104.97秒／最大RSS1,757,648KiB（約1.68GiB）。準備180秒＋終了120秒のCPU予約を含む保守計上404.91秒で1200秒上限内。この予約値は実測CPUではなく、個別の実測receiptを別途保持した。全体保全の実CPUは開始26.00秒・終了24.69秒。新runは約25MB、空き約66.8GB。Goal開始から保存時点までのwallは [RESOURCE_CLOSURE.json](RESOURCE_CLOSURE.json) と最終publication receiptに残す。上限1800秒・run1GiB・空き10GiBを維持し、上限変更・再試行は0。

新download/install/研究API/モデルAPI/キー照会は全て0。Git保存だけ通信する。旧960 slots/934 success/25出力失敗/HTTP520欠測1、991 historical calls、不明usage予約2、元20比較、正常empty successは不変。既存Draftへの操作は0で、旧保存receiptを参照する。`main` は `7309327efd1c19dda793b19e8d8e6ce0b5868db9` を保持する。

実行したCLI（同じrunへの再実行はexclusive-output guardで拒否する）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TMPDIR=/root/minecraft-runs/goal014-20261017T145709Z/tmp \
/root/minecraft-tools/goal013a-20261017T131846Z/venv/bin/python -B \
  -m offline_evaluation.analysis.goal014.run_pipeline \
  --config /root/workspace/Minecraft-view-dataset/reports/server_only/goal014/CONFIG.json \
  --plan-commit 6c7019094b0afdfe28ee88d5cc086797c62e457a
```

個別fixtureは同じPython・thread・bytecode設定で `-m pytest offline_evaluation/analysis/goal014/tests/test_diagnostics.py -q -p no:cacheprovider --basetemp <新run>/tests/diagnostics`（statistics/binding/pipelineも対応するファイル名）で実行できる。実データ再実行が別途必要になった場合は新runとそのstart時刻・保全snapshotを新configへ束縛し、そのcode/plan/configをcommitして `--plan-commit` に渡す。既存runを削除して再実行せず、現在の一巡制限を自動解除しない。新依存や重み取得は不要。

要件対応は [REQUIREMENT_COVERAGE.json](REQUIREMENT_COVERAGE.json)。小さいコード・報告を専用branchへnon-force pushし、remote/HEAD/cleanの実確認は原本runのPUBLICATION_RECEIPTに保存する。実装parity、固定gallery識別信号、Minecraft人間品質妥当性未検証、RQ2限定版、原RQ2未検証を区別して、この12項目で終了する。論文・次Goal・新Builder実験は作成しない。
