**Goal013A：固定scoreと保存O再構築outcomeの関連解析仕様**

位置付けは `POST_HOC_EXPLORATORY / GT_INDEPENDENT_SUMMARY_ALIGNMENT`。Goal013の[SCORING_PROTOCOL.md](../goal013/SCORING_PROTOCOL.md)のPhase 2を継承する。これはpreregistrationではない。旧研究outcomeは既知であり、全担当者が完全にblindだったとは主張しない。Goal013Aの実得点や新しい相関を用いて本仕様・checkpoint・prefix・集約法を変更しない。

この仕様作成・合成fixture検証時点では、Goal013Aの実得点計算、実outcome結合、新しいassociation計算は未実行。ここにある定義やfixture数値は実研究結果ではない。コードと実測hashを含む採点仕様を別commitで固定し、モデル検証・採点・score凍結・代表性gateが通った後だけ、root実行系が以下へ進む。旧Goal013の終了状態は変更しない。

**実outcomeを開く前の境界**

`outcome_binding.load_frozen_outcomes(freeze_path, expected_freeze_sha256=...)` は、呼出し側が先に記録したfreeze receiptのSHA256に一致する原本だけを読む。次の全条件を満たすまで、Goal007の最終outcome artifactを開かない。

1. freeze receiptの `status`、`source_binding_status`、`scorer_validation_status` がすべて `PASS`。
2. 唯一の `score_id` は `MEAN6_CLIPSCORE_VITB32_CPU_FP32_SUMMARY_ONLY_GOAL013A`。`code_commit` と `spec_commit` は異なる40桁Git commit ID。
3. `representativeness_artifact`、`score_artifact`、`source_binding_artifact` は `{path, sha256, bytes}` を持ち、ローカル原本の全bytesが一致する。symlink、相対path、traversal、非有限JSON、重複JSON keyを拒否する。
4. representative JSONの `status`、`overall.status` と `providers.{openai,anthropic,gemini}.status` が全て `PASS`。`FAIL`、`INDETERMINATE`、欠落は停止する。
5. sourceとscoreは同じ固定60建築×3 providersの全180行を持ち、重複・脱落がない。各行の `scene_id`、`description_provider`、`O_input_id`、`O_input_sha256`、`description_source_sha256`、`summary_sha256` が一致し、O inputのbyte hashと元descriptionのbyte hashも一致する。`images_ordered` の6個の `{path,sha256,bytes}` は順序ごと一致し、6pathは互いに異なる。原summary欠測行にscoreを補わない。

score/source JSONは行listまたは `{rows: [...]}`。各score行は上のidentity fieldsと、`score_id`、`score_status`、`score`、`token_coverage_status`、`summary_token_count_untruncated` を持つ。`COMPUTATION_VALID` は有限scoreを要し、他の計算statusはscore `null`。Phase 2適格にはさらに `token_coverage_status='VERIFIED'` と正の非切捨てsummary token数を要する。token列を検証できない行は、scoreを算出できても関連解析へ入れない。

代表性の閾値はGoal013から変更しない。全体またはいずれかのdescription providerについて、非空summaryの過半数が未読token率 `>0.5` ならPhase 2を停止する。token診断の未確認行がある場合の保守的判定はMETHOD_SPECで採点前に固定し、`INDETERMINATE` はこのloaderで通さない。空summaryを分母の適格行やscore 0へ変換しない。

**結合元と原本同一性**

Goal007の公開MANIFESTの固定SHA256は `ef05c2ed93247802e9a68a00d700d39a8934d6231c7787315ab37b01d2467714`。その `output_root` が指すcanonical MANIFESTも同じSHAを要する。このmanifestの `products['data/final_slots_960.json']` が指定するbyte hash・sizeからcanonical selected slotsを読む。現在固定されたproduct SHA256は `f0960ef78f21bd42c78cfeb709cf9505bbf34d27f60a4986824ea763cc061b2d`。ファイル名だけから別のrunや類似scoreを探索しない。

最終960 slots、934 success・25 failed・1 operational missingという固定scopeを保持し、その中の `condition='O'` 360 selected slotsだけを結合する。keyは建築×description provider×Builderに、O input ID・O input/source byte SHAの一致を加えたもの。Builderは `openai` と `anthropic` を区別する。selected attempt IDとselected-final flagを要し、attempt履歴や承認済みretryを追加の観測へ増やさない。

使用outcomeは保存 `server_v2.2_strict_translation` の `metrics.aligned.occupancy_iou` に固定する。GT・voxelを開かず、alignment再探索・再建築・metric再計算をしない。source hash確認のために読むmetadataとモデル入力を混同しない。結合結果はscoreと保存scalar・最小限のselected-source provenanceだけをallowlistで保持し、GT、voxel artifact、raw response、usage payloadは出力しない。

**主探索A：pooled Spearman**

OpenAI Builder、Anthropic Builder、両Builder同時validな同一説明のIoU算術平均、の3対象。有限な適格summary得点とoutcomeの共通集合をpoolし、丸め前値の完全一致による平均順位同士のPearson相関を求める。provider別順位化、標準化、事後epsilon、p値は使わない。対応行2未満、いずれかの順位vectorが定数、算術未定義の場合は `null`。

表にはplanned 60建築・180説明、実paired description行数、寄与建築数、provider別の実対応行数・建築数、failure0補完行数・Builder slot数を残す。180、178、60をactual association Nとして流用しない。両Builderを独立360観測としてpoolしない。点推定は説明行をpoolするため欠測によって建築当たり寄与数が異なる。建築cluster CIを使っても点推定が建築等重みになるわけではない。

**主探索B：建築内pairwise concordance**

各Builder別に、各建築の3 description providersから3 unordered pairsを作る。全60建築で計画180 pairs。まず片方でもscoreまたはoutcome不適格のpairを `missing_pairs` に分類する。両側観測済みpairでは、得点のみtie、outcomeのみtie、両方tieを排他的に数え、全て分母から除く。tieは丸め前値の完全一致で、0.5点付与や乱択をしない。

残るpairで得点差とIoU差のsignが同じならconcordant、逆ならdiscordant。建築内値は `C_b = concordant_b / (concordant_b + discordant_b)`。比較可能pair 0の建築は `null`。全体は非nullの `C_b` を建築等重みで算術平均する。全pairをpoolした率へ置き換えない。両Builder平均のconcordanceは追加しない。

各表に実適格description行数、寄与建築数、planned pairs、missing、得点のみtie、outcomeのみtie、両tie、比較可能、concordant、discordantを保存し、建築別の同じ内訳も残す。missing＋3種類のtie＋比較可能がplanned pairsに一致し、concordant＋discordantが比較可能に一致する。AとBは異なる問いの推定量である。

**valid-onlyとknown failure 0の区別**

主表は `valid_only`。別感度表 `known_failure0` では、selected O slotが `failed`、usage-known、完了済みoutcomeであり、既知failure kindが `parse_failure`、`validation_failure`、`execution_failure`、`truncation` のいずれかである場合だけ、IoUを一時的に0として扱う。401/520運用障害をこの集合に入れない。原本を変更しない。

独立scoreの欠測・推論失敗・token coverage未確認、成功出力で未定義のscalar、運用欠測、その他の不明failureは0にしない。正常empty successに保存されている有限IoU 0は通常のvalid観測のままで、model failureへ再分類しない。両Builder平均のfailure0表は、両側がvalidまたは識別済みfailure0対象の同一説明だけで計算する。HTTP520のglobal942はJの欠測であり、Oへ移さない。

**固定cluster bootstrapとCI**

planned 60 building IDsを昇順にして、`NumPy Generator(PCG64)`、seed `20261017`、2000 draws、各draw60個の復元抽出とする。同じindex行列を全3対象×2 policies、2 Builders×2 policies、後述4補助診断に共有する。行列のlittle-endian int64 bytesのSHA256、NumPy版、seed、draw数を保存する。

欠測建築をplanned frameから先に落とさない。抽出された同一建築の多重度を保持する。Aは複製した全適格説明行をpoolし、drawごとに順位を再計算する。Bは保存した非null `C_b` を抽出多重度付きで平均する。建築IDで重複を再統合しない。

未定義drawは `null` として保存・計数し、補充しない。元の寄与建築2以上かつ2000中1900以上の有限drawがある場合だけ、有限drawの2.5/97.5 percentileを探索的95% CIとする。補間はHyndman–Fan type 7、NumPy `method='linear'`。それ以外は点推定と有限/未定義draw数を残し、CIを `null / INSUFFICIENT_BOOTSTRAP_VALIDITY` とする。これは有限drawに条件付けた区間であり、1900は運用上の安定性gate。p値、significance label、Holm family、CIを見たdraw増加は行わない。

**固定補助診断と主張の限界**

適格scoreと非切捨てsummary token長のSpearmanだけを、全provider pooledおよび3 providers別の計4表として出す。outcomeの有無でこの診断の対象を削らない。tie・欠測・planned60 cluster・2000 draws・CI gateはAと同じ。provider別outcome相関や新しい難度指標、GT proxyとの関連を追加しない。6視点min/max/population SD、token/crop coverageは採点側の記述に留める。

summary限定scoreとJSON全文を読むBuilderは入力情報の範囲が異なる。低相関は説明全文の有用性を否定せず、高相関も因果効果、独立外部予測性能、人間品質尺度の妥当性を証明しない。GT非参照という採点入力制約は、画像や建築難度との統計的独立を意味しない。原RQ2・RQ3・RQ4の判定は自動的に変更しない。

実装は `offline_evaluation/analysis/goal013a/{outcome_binding,associations}.py`。合成fixtureは `tests/test_associations.py`。NumPyだけを用い、SciPyを導入しない。fixtureでは独立なlower/equal-count順位法、建築等重みとpair重みの差、抽出多重度、tie排他性、failure0境界、CI gate、hash不一致やgate停止時のoutcome未読を検証する。fixture成功を実モデルの妥当性や実研究結果と扱わない。
