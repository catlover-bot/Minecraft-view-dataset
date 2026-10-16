# Goal010 — 研究妥当性の監査とCPU対照

**中心主張には修正が必要である。O説明の寸法だけから作った充填直方体が、両LLM Builderの平均aligned occupancy IoUを上回った。** これは占有IoUについての反証であり、材質・意味的構造・見た目まで箱が優れているという結果ではない。「LLM再建築の優越性」より、どの情報・評価・実行上の失敗が結果を左右するかを検証する診断研究へ主張を絞る必要がある。

全6PhaseをCPUで実施した。Phase4は監査を完了したが、投影と遮蔽の検証不足により可視性の数値評価を実施していない。今回の研究API生成、モデル照会、token-count API、キー検査、人手実験は全て0件。旧成果物・予算台帳・Draft Releasesは変更していない。本文は論文原稿ではなく、研究判断のための監査報告である。

## 結果の要点

| Phase | 実施結果 | 研究上の意味 |
|---|---|---|
| 0: 出典・入力 | 180 request、60建築、全件実入力6画像／保存候補12画像。scene内3 providerで画像bytes・順序・prompt一致 | 画像差は今回のprovider差の説明にならないが、モデルversion・出力上限・時期は交絡している |
| 1: 幾何baseline | 480予測をGT評価前に保存、480件の厳密評価と独立検算一致。Pの密度恒等式60件一致 | Oでは寸法だけのfilled-boxに対するLLMの主要占有指標の優越性を支持しない |
| 2: provider対応比較 | 固定36対比、各57–60独立建築。2,118の指標別pair、42の指標別除外を保存 | 説明生成元とBuilderを区別した「設定込みの差」。固有能力・表記だけの因果効果ではない |
| 3: E・材質 | 60/60 CPU control完全一致。101 valid結果を独立再計算。E truncationは5/60・14/60 | 占有一致でも材質転記誤りが残る。CPU controlはLLM成功に数えない |
| 4: 可視性 | `NOT_VALIDATED_NO_VISIBILITY_CONCLUSION` | 「不可視領域が主原因」「6枚で何%可視」とは言えない |
| 5: proxy | 固定1,350行全て保存。1,230推定可能、60定数、60 CI不安定 | 寸法成分への感度を確認したが、説明品質の妥当性や独立因果寄与は証明していない |

全て **POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC**。従来の結果を見た後の分析であり、事前登録とは呼ばない。[計算前仕様](posthoc_specification.md)をcommit `87f85a2b6bc06a1ae30af8e5276d3aff67c58379`で固定し、その後に実データ解析を行った。新しいp値や有意性ラベルは付けていない。

## Phase0: 何が同じ条件だったか

100建築のGTサイズ順位を5層に分け、層内で固定乱択した60評価・10 smoke・30 restを再確認した。独立unitは60建築で、180説明や360 Builder観測ではない。既存データの部分集合であり、完全未見データとは呼ばない。

全180件の保存request記録で6画像を参照し、保存候補は12画像だった。画像・promptは同一sceneの3 providerで一致する。一方、説明モデルはOpenAI `gpt-5-mini`、Anthropic `claude-haiku-4-5-20251001`、Gemini `gemini-3.1-pro-preview`、max_tokensは1800/1800/4096、temperatureは全件0.2。保存記録時期はそれぞれ2026年2月末・3月初・4月中旬である。これは過去の保存設定の記述で、現在のモデル提供状況を照会したものではない。`created_at`は応答後の記録作成時刻で、API送信開始時刻ではない。

Oは人間可読フィールドを含む保存JSON、Jは同じtyped treeの再シリアライズである。O/Jは別々の単発生成で、Builderのseed/temperature/top_pを明示固定した純粋な表記実験ではない。PはGTの部分属性、実際のEはsorted paletteとinclusive cuboidsの完全離散仕様である。storage bbox、occupied bbox、撮影generation originを分け、NPYのYXZを既存safe readerでworld XYZへ変換した。

[詳細解釈](PROTOCOL_INTERPRETATION_JA.md)、[監査](analysis/phase0_protocol/PROTOCOL_AUDIT.md)、[source bindings](analysis/phase0_protocol/SOURCE_BINDINGS.json)、[入力・評価の補足](analysis/phase0_protocol/PROTOCOL_EVALUATION_SUPPLEMENT_JA.md)。

## Phase1: 単純な箱を超えたか

O baselineは説明の`dimensions_estimate`だけを受け取り、暫定原点(0,0,0)に構築した。width→X、height→Y、depth→Z、正の数値を固定のround-to-even規則で整数化する。P baselineは保存P入力の`world_bbox`だけを使う。Oの構築関数はGT座標・占有配列を参照せず、Pも開示world_bbox以外のvoxel_count・材質・style・個別voxel配置を構築に使っていない。入力別の最適化は行わず、充填箱と厚さ1の六面shellを両方保存し、人工材質のスコアは論証に使わない。

以下は同じ入力のcommon-valid比較。差はBuilder−baseline、95% CIは2,000回の建築cluster bootstrap、seed20261017。Oは各建築内で適格説明の差を平均してから60建築を等重みにする。各Builderは同じ説明を使う反復測定であり、両者のNを合算しない。

| 入力・形状 | Builder | 独立建築 / input pairs | Builder平均 | baseline平均 | 差 [95%探索的CI] |
|---|---|---:|---:|---:|---|
| O filled-box | OpenAI | 60 / 177 | 0.17603 | 0.18799 | −0.01196 [−0.02035, −0.00323] |
| O filled-box | Anthropic | 60 / 179 | 0.17717 | 0.18878 | −0.01161 [−0.01909, −0.00450] |
| O shell | OpenAI | 60 / 177 | 0.17603 | 0.11939 | +0.05664 [+0.04767, +0.06638] |
| O shell | Anthropic | 60 / 179 | 0.17717 | 0.12010 | +0.05707 [+0.04895, +0.06562] |
| P filled-box | OpenAI | 60 / 60 | 0.16736 | 0.14838 | +0.01898 [+0.00538, +0.03300] |
| P filled-box | Anthropic | 60 / 60 | 0.16909 | 0.14838 | +0.02072 [+0.01398, +0.02773] |
| P shell | OpenAI | 60 / 60 | 0.16736 | 0.10858 | +0.05877 [+0.04835, +0.07022] |
| P shell | Anthropic | 60 / 60 | 0.16909 | 0.10858 | +0.06051 [+0.04893, +0.07298] |

Oの既知出力失敗3件/1件をIoU=0とする補助解析でも結論は反転しない。baseline自身の入力・CPU失敗は0件。未知のoperational missingは0補完しない。全80比較、4種の条件・構造、absolute値、全建築と5サイズ層の分布を保存した。aligned評価はGTを使った位置合わせであり、絶対配置の成功を意味しない。Oのabsolute IoUが0でも、暫定原点に起因する不一致と形状不一致を混同しない。

独立検算は3次元累積和で全交差候補の箱/shell占有を厳密計算し、既存の厳密平行移動評価と最大交差数・同点shiftを照合した。P filled-boxのabsolute IoUは60件全てで`N_GT / P_bbox_volume`に一致した。building034は4912/40320=0.1218254で、この例のstorage/occupied bboxは一致する。

O箱は形状・材質をほぼ表現しない弱い構造対照だが、それを主要占有指標で上回れなかったことは無視できない。Pは正解bboxを持つ強い情報対照であり、通常Oと混ぜたランキングにはしない。結果は「LLMの能力全てが箱以下」ではなく、「現在の主要占有指標だけで複雑な再建築の付加価値を示せない」と解釈する。

[詳細と全指標](BASELINE_INTERPRETATION_JA.md)、[80比較CSV](analysis/phase1_baselines/baseline_matched_deltas.csv)、[480予測評価](analysis/phase1_baselines/baseline_per_building.csv)、[サイズ分布](analysis/phase1_baselines/baseline_size_distributions.csv)。

## Phase2: 説明生成元の差

Oのaligned occupancy IoUで、Gemini−OpenAI説明の差はOpenAI Builderで+0.01813 [−0.00162, +0.03881]、N57、Anthropic Builderで+0.00386 [−0.01368, +0.02309]、N59だった。Gemini−Anthropicは+0.06993 [+0.04803, +0.09276]と+0.06198 [+0.04035, +0.08300]、各N60。Anthropic−OpenAIも含む全36対比を掲載し、J、寸法MAE、材質IoUを選別せず残した。

これらは同じscene・Builderのcommon-valid説明をペア化した記述で、設定込みの差である。固有モデル能力や因果的な交互作用を識別していない。baselineのplanned-frame bootstrapと異なり、この対比のbootstrap frameは共通valid建築である。

[全36対比と除外](FACTOR_FINDINGS_JA.md)、[対応比較CSV](analysis/phase2_factors/description_builder_matched.csv)、[設定交絡](analysis/phase2_factors/description_builder_confounds.json)。

## Phase3: 占有が正しくても材質は誤る

GT→固定E→参照actions→実行の完全一致は60/60だった。LLM E結果はOpenAI55 success・5 truncation、Anthropic46 success・14 truncation。valid101件で保存shiftを固定した独立再計算が元のabsolute/aligned strict値と一致した。CPU controlは実行系の検証であり、LLM成功ではない。

building034のOpenAI Eは4,912セルの占有が全て一致し、寸法MAE=0だったが、材質一致は1,794、置換は3,118。材質IoUは1794/(4912+4912−1794)=0.2234122である。GT glass→予測brick_blockが160セル、sandstone→glassが2,958セル。参照428 cuboidの境界は全て出力に現れ、正しいlabelを持つ境界は202だった。位置形状が正しいこととpaletteの正しい転記は別問題である。

事前に固定した狭い歴史的alias規則は101件いずれのaligned材質IoUも変えなかった。これは全ての意味的等価性を否定する検査ではないが、今回のaliasで元の誤りを解消する根拠はない。occupancy>=0.99かつ材質IoU<0.5の8例も全て保存した。出力上限32768は元plan由来で、参照actionsのbytesをtoken数へ換算せず、truncation後の完成出力を推測しない。

[材質・容量の詳細](MATERIAL_FINDINGS_JA.md)、[混同行列CSV](analysis/phase3_exact_oracle/E_material_confusion.csv)、[全120 outcomes](analysis/phase3_exact_oracle/E_error_taxonomy.json)、[容量診断](analysis/phase3_exact_oracle/E_capacity_diagnostics.csv)。

## Phase4: 可視性を原因と呼べるか

720保存画像は960×540、記録FOVは70度。ただしFOVはrequested hintで、実投影を保証しない。actual XYZがあるのは658 viewsで、actual yaw/pitchは保存されていない。光学中心・eye offset・遮蔽・ブロック形状/透明性も実RGBとの独立照合で検証されていない。

したがって可視・不可視割合、6対12枚の可視差、領域別IoUは計算していない。`visibility_region_metrics.csv`を作らなかった理由は検証不能であり、欠けた観測を合成画像で補っていない。「不可視領域が主原因」とする説明は却下する。

[検証可能性JSON](analysis/phase4_visibility/VISIBILITY_FEASIBILITY.json)、[限界](analysis/phase4_visibility/VISIBILITY_LIMITATIONS_JA.md)。

## Phase5: proxyは何を測っているか

972層別proxy相関、108建築内関連、270共変量との関連、計1,350行を保持した。1,230行はCI付きで推定可能、60行はproxy定数、60行は2,000 drawsのうち必要な1,900有限drawを満たさずCIなしである。成功した行だけを抜き出した全成功表示にはしていない。

全体valid-onlyの点推定は元Goal008と同じで、compositeのρは0.39317/0.37449、dimensions-onlyは0.56974/0.54525。新しいseedで計算したdimensions-onlyの95%探索的CIは[0.46689,0.66225]/[0.42664,0.64647]、各60建築、177/179 pairs。寸法成分を除く固定算術感度ではρが0.20590/0.21282となる。しかし共有GT、建築規模・密度・provider、valid選択があり、寸法の独立因果寄与を推定したわけではない。

元の180 scores・360分析は不変で、従来RQ2は`NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES`のまま。新しい説明品質尺度を学習・検証したとは言えない。

[全結果の解釈](PROXY_FINDINGS_JA.md)、[1,350行CSV](analysis/phase5_robustness/proxy_robustness.csv)、[全設定・分母・層境界JSON](analysis/phase5_robustness/proxy_robustness.json)。

## 検証・保存・再実行

- 終了した解析コードのテスト269件（新規Goal010 182件＋既存core/alignment/oracle/cuboid 87件）、コピー処理20件、計289件PASS、fail/error/skipは0。開発途中の失敗記録も保持している。初期fixtureのOS socket生成失敗は、実socketを作らず接続禁止hookを検証する形に直した。過去成果物やOS設定を変えてテストを通してはいない。
- 全6Phaseの実時間合計378.915秒、CPU合計377.968秒、最大RSS557,376KiB（約544.3MiB）。解析テストCPU8.47秒、コピー処理CPU23.43秒は別計測。これは計測対象コマンドの値で、対話・実装・Git通信を含む作業全体時間ではない。
- Python3.12.3、NumPy2.5.4、SciPy1.18.1。既存venvを使用しBLAS/OpenMP/MKL各1 thread。Phase5終了時空き69,383,176,192 bytes。
- 原本960/934/25/1、961固定attempts、991歴史的calls、401→940選択retry、942のnull、unknown-usage予約2件、Goal007の20比較、Goal008の180/360とコードhashは全て不変。過去USD21.56083625は保守的予算計算値で請求書ではなく、今回の追加生成費用ではない。
- root接続を維持。実効設定workspace-write / auto_review。通常sandboxコマンドが実行でき、必要なrun/Git書込みのみ対象別承認を使用した。OSレベル隔離は保証していない。`hostname -I`は`Operation not permitted`で、アドレスは再認証できていない。サンドボックス起動エラーは観測していない。[実行環境記録](execution_environment.json)。
- 原本を上書きせず `/root/minecraft-runs/goal010-20261016T223832Z` に保存。大きな予測座標/actionsはこのrunだけに保全。42集計/receiptファイルを[analysis/](analysis/)へコピーし、全bytes/SHA一致を検証した。[コピー証明](analysis/PUBLIC_COPY_RECEIPT.json)。原model responseやGT全rawを新規Git保存していない。
- 既存Draft Releases414157245/414396750/414429861の18 assetsをGET metadataで照合し、ID/サイズ/GitHub digest/draft/target一致。新規ダウンロード・upload・Draft更新・publishは0。[読取り照合証拠](existing_drafts_readonly_verification.json)。

仕様commitは`87f85a2b6bc06a1ae30af8e5276d3aff67c58379`、解析コードcommitは`4a3b714ad139ee0f2836d756347b6bf02dbb9a8b`、コピーコードcommitは`a68505607ac5775f245560fe7dce9eba5efb6828`。保存先branchは`research/server-only-20261013`で、main基準は`7309327efd1c19dda793b19e8d8e6ce0b5868db9`。最終保存commit/remote一致はこの報告を含むcommit後に別途確認する。

再実行は新しいUTC run名を選び、以下の`--phase`を0→1→2→3→4→5の順に実行する。既存出力先への再実行は拒否される。原本hashやcode/spec bindingの不一致でも停止する。追加の生成を起動するコマンドではない。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-server-v2-20261013/bin/python -B -m offline_evaluation.analysis.goal010.pipeline \
  --phase 0 --run-root /root/minecraft-runs/goal010-YYYYMMDDTHHMMSSZ \
  --code-commit 4a3b714ad139ee0f2836d756347b6bf02dbb9a8b \
  --specification-commit 87f85a2b6bc06a1ae30af8e5276d3aff67c58379
```

テストは`python -B -m pytest -q offline_evaluation/analysis/goal010/tests`で実行できる。詳細な研究判断と、実行していない最小追加実験案は[RESEARCH_DECISION_JA.md](RESEARCH_DECISION_JA.md)にまとめた。
