Goal013Aは完了した。公式OpenAI CLIP ViT-B/32を専用CPU環境へ導入し、著者実装との実モデル一致検証後に178/180 summary、1068/1080画像対を採点した。全代表性gateがPASSし、固定したBuilder O結果との関連分析まで実行した。旧Goal013の採点0という終了記録と、旧RQ2が未検証である状態は変更していない。

主結果（valid-only）は以下。記述行数と独立な建築cluster数を区別する。

| Builder target | 実paired行 | 寄与建築 | Spearman rho | 探索的95% CI |
| --- | ---: | ---: | ---: | --- |
| openai | 175 | 60 | -0.07692 | [-0.22561, 0.06731] |
| anthropic | 177 | 60 | -0.12389 | [-0.26488, 0.01379] |
| both_builders_mean | 174 | 60 | -0.10831 | [-0.25426, 0.02977] |

いずれの区間も0を含む。建築内順位一致率はOpenAI 0.44633（寄与59建築、CI [0.36207,0.52874]）、Anthropic 0.46111（60建築、[0.38333,0.53889]）。全数値・既定failure0感度・token長補助診断・tie/missing分母は [RQ2_EXPLORATORY_RESULTS_JA.md](RQ2_EXPLORATORY_RESULTS_JA.md) と [ASSOCIATIONS_SUMMARY.json](ASSOCIATIONS_SUMMARY.json) に保存した。14 estimandsすべて2000/2000 finite cluster draws、seed20261017、type7 CI。p値・新Holm family・新subgroup探索は0。

**導入した環境と真正性。** 新規取得と専用venvへのinstallは今回の明示許可に基づく。過去のdownload禁止の範囲変更を [START_STATE.json](START_STATE.json) に記録した。生成・embedding・judge・model一覧・token-count等の研究API送信は0、新規Builder生成や人手実験も0。

tools rootは `/root/minecraft-tools/goal013a-20261017T131846Z`、venvはその`venv/`、run rootは `/root/minecraft-runs/goal013a-20261017T131846Z`。公式著者実装commit `1036465276513621f77f1c2208d742e4a430781f`、OpenAI CLIP commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`、公式Azure checkpointを使用した。checkpoint実測353,976,522 bytes、SHA256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` は公式expectedと一致。全bytesを取得後にhashした。

Python3.12.3、torch2.9.1+cpu、torchvision0.24.1+cpu、numpy1.26.4、Pillow11.3.0、ftfy6.3.1、regex2025.9.18、packaging25.0等22wheelを全hash固定のoffline installで導入。pip check PASS。全実版・配布URL・wheel/BPE/source/LICENSE hashは [DEPENDENCY_MANIFEST.json](DEPENDENCY_MANIFEST.json)、[requirements.lock](requirements.lock)、[SCORER_CONFIG.json](SCORER_CONFIG.json)。独立取得監査は [ACQUISITION_AUDIT.json](ACQUISITION_AUDIT.json)。ソースはMIT LICENSE、公式checkpointの利用条件は同repo LICENSEと研究用途model-cardに基づく確認で、別のweight固有license文書を取得したとの主張はしていない。

取得通信は **57 HTTP requests / 584,832,575 response-body bytes**（metadata/indexを含む、redirect0、HTTP200が57件）。TLS/header/wire overheadと文献ブラウザ閲覧はこのbyte計測外。研究API送信0回と区別する。途中の3つのローカルresolverエラーは記録し、取得済みbytesを再hashして再利用した。追加mirror、別モデルへのfallback、重み再downloadはない。再実行用取得adapterのredirect計数・LICENCE表記・path guardの補強は監査へ記録し、元取得receiptを書き換えていない。

**採点と検証。** score IDは `MEAN6_CLIPSCORE_VITB32_CPU_FP32_SUMMARY_ONLY_GOAL013A`。prefix `A photo depicts `、CPU float32、q=2.5×max(dot(normalized v,t),0)、6枚の算術平均。原著者の224 bicubic→center crop→RGB→tensor→normalize。最高視点、feature平均、0–1上限clip、logit scalingは使用していない。model入力は原summaryと6画像だけ。

原著者ASTから必要な5定義をそのまま抽出し、元extractors/get_clip_scoreを各1回実行した。実token・tensorは完全一致、normalized featuresとqの最大差0。batch1/8の最大feature差1.490116×10⁻⁷、独立scalar式との差5.523586×10⁻⁸で事前許容差1e-5内。cache replayと6視点順序不変もPASS。2021年GPU/float16とのbit一致は主張しない。[REFERENCE_PARITY.json](REFERENCE_PARITY.json) と [INDEPENDENT_PARITY_REVIEW.json](INDEPENDENT_PARITY_REVIEW.json) を参照。

parity単体CPU5.05秒/経過5.07秒。先頭1建築pilot CPU1.53秒、先頭10建築pilot15.10秒（30行）、安全倍率込み全量予測185.20 CPU秒で予算内だった。同じプロセスで60建築へ進み、360画像と178textを各一度encodeした（parityの反復は別）。全採点までの累積CPU109.64秒/経過110.15秒、最大RSS1,746,452KiB（約1.67GiB）。source前後照合等を含む採点プロセス全体はCPU145.50秒/経過146.08秒。関連分析プロセス全体はCPU44.86秒/経過44.88秒、RSS144,440KiB。設定はbatch≤8/workers0/thread1、CPU3600秒/wall7200秒、RSS上限6GiB（目標4GiB）。全て上限内で終了した。Python guardとlockはOS隔離保証ではない。

非空178件は全件COMPUTATION_VALIDでtoken coverage VERIFIED、計算失敗0。Gemini building_054/061の空summary2件・12画像参照は欠測のまま。切捨て59件（OpenAI38/60、Anthropic14/60、Gemini7/58）。prefix3token、summary容量72token、原summary長38–127token、最大未読率43.307%。未読率>50%は全体/各providerとも0でgate PASS。実77token IDsはcanonicalのみ保存し、公開版はhash/count/maskを保持する。[SCORES_180.csv](SCORES_180.csv)、[TOKEN_COVERAGE_180.csv](TOKEN_COVERAGE_180.csv)、完全なsource対応を含む [SCORES_180.json](SCORES_180.json) を参照。

360画像はすべて960×540→398×224→224×224、残る面積56.28%。建築全体がcrop内にあるとは推測しない。固定scene順001/003/005のfirst view cropだけcanonicalへ保存した。元画像とcropはGitへ登録していない。[CROP_METADATA_360.json](CROP_METADATA_360.json) に出典と幾何を保存。

**固定順序と保全。** 最終code commit `de7c1cf3c6ddaf1f2b84ed467375bbf4de805642`、別spec commit `ded60648d0e0cb9a7892bef451e8696657d4e58b` を実モデル計算前に固定した。最初のcode/spec pair後に4つの境界fixtureが到着したため、履歴を変更せずもう一組を追加した。方法・結果を見た調整ではない。最終新規fixture **152 PASS / fail0 / error0 / skip0**。全過去テストの再実行はしなかった。

原本の終了hash確認後にscore freeze SHA `b014460fc3119c2ba6066f7e87d4bf5ba09cef3390f4be57c47729cb205b3ae0` を保存し、その後の別stageで関連分析した。canonical full bootstrap artifact SHA `b47b4984fd85af36ecb885956e960bf7b91a537ad0bb3c1381e7d5916e584049`。sourceの期待SHA、順序、token、数値、freezeを独立確認した [INDEPENDENT_SCORE_REVIEW.json](INDEPENDENT_SCORE_REVIEW.json) を参照。関連分析の独立検算は [INDEPENDENT_ASSOCIATION_REVIEW.json](INDEPENDENT_ASSOCIATION_REVIEW.json)（点推定の最大差5.56×10⁻¹⁷、CI差0）。公開copyは [PUBLIC_COPY_RECEIPT.json](PUBLIC_COPY_RECEIPT.json) でcanonicalへ束縛する。

[INTEGRITY_AFTER.json](INTEGRITY_AFTER.json) は旧28,329 files、旧tracked18,273、既存venv16,477 files/links、cache6filesと欠如roots、Goal013 canonical tree、mainが開始snapshotと同一と確認した。14 read-only locks、旧領域への書込み・研究API通信attemptは両stageとも0。960 slots/934 success/25モデル失敗/HTTP520欠測1、991 historical calls、不明usage予約2、承認済global940 retry、正常empty successは不変。既存Draftへの操作は0で、旧Goal013の保存verification receiptを参照した。今回のfresh Draft通信やasset復元証明ではない。

今回進んだのはsummaryの画像整合度に限る。BuilderはO JSON全文を読むため情報範囲が異なり、低相関を全文品質の無効性へ一般化しない。GT非参照は統計的独立を意味せず、人間caption品質・外部予測・因果的情報損失・美観/機能は未検証。現在の結果を保存して終了し、追加の有料生成や指標探索へ広げない。

実行手順は [EXECUTION_SPEC.md](EXECUTION_SPEC.md) と [FINAL_EXECUTION_BINDING.json](FINAL_EXECUTION_BINDING.json)、実際の全CLI引数は各PROCESS_RESOURCES.txtに残した。score stage成功の後、同じcode/spec引数で `associate --freeze-sha256 b014460fc3119c2ba6066f7e87d4bf5ba09cef3390f4be57c47729cb205b3ae0` を実行した。出力はexclusiveで同じrunを上書きしない。再実行時は新専用root・その実測configを別commitで固定し、このGoalの累積資源使用を引き継ぐ。

成果commitはこのREADMEとMANIFESTを保存するcommitとしてGitから解決できる。専用branch `research/server-only-20261013` へnon-force push後にremote HEAD/cleanを検証し、最終応答で実SHAを報告する。mainは `7309327efd1c19dda793b19e8d8e6ce0b5868db9` のまま。
