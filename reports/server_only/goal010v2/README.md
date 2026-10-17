# Goal010 v2 — 占有・表面・材質を分けるCPU指標監査

**O入力では、単純な充填箱は占有IoUとRecallで優位、LLM BuilderはPrecisionと表面F-scoreで優位だった。** 元の占有IoUに関する反証を維持しつつ、「箱が全面的に優れる」という解釈も退ける。指標は異なる構成概念を測っており、表面F-scoreの改善を画像の見た目や意味構造の忠実性まで一般化できない。

旧Goal010は完了済みのため、本v2はSection 1.5の指標監査を別コード・別runで追加する。旧Goal007の20比較、Goal008の180 proxy・360分析、旧Goal010の全6Phaseは不変であり、新しい数値で置換しない。新解析のラベルは **POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC**。結果を既に見た後の追加分析で、事前登録ではない。

## 今回の範囲と再利用した監査

| 項目 | 今回の処理と証拠 |
| --- | --- |
| 出典・固定cohort | 60建築・960枠のsource binding、元attempt、input/GT/prediction SHAを各段階の前後で再検証。[新SOURCE_BINDINGS](analysis/0_source/SOURCE_BINDINGS.json) |
| 新指標の規約 | [旧指標監査](METRIC_AUDIT_JA.md)、[指標仕様](METRIC_SPEC.md)、[registry](METRIC_REGISTRY.json)、[統計仕様](STATS_SPEC.md)を実データ計算前のcommitで固定 |
| 全成功予測とbaseline | 保存済みの934成功予測と480固定baselineを読取り専用で計測。失敗・欠測26枠もNULLと理由を持つ行として保持 |
| 入力・撮影request | [旧Phase0監査](../goal010/analysis/phase0_protocol/PROTOCOL_AUDIT.md)を再利用。180 requestすべて実入力6画像、保存候補12画像。scene内3 providerで画像bytes・順序・prompt一致 |
| provider対応比較 | [旧36対比](../goal010/FACTOR_FINDINGS_JA.md)を再利用。モデルversion・出力上限1800/4096・生成時期を含む設定込みの差 |
| E・材質・容量 | [旧E解析](../goal010/MATERIAL_FINDINGS_JA.md)を維持し、新しいM/C、M/g、macro recallを追加。容量診断は保存120 outcomesを可視化 |
| 可視性 | [旧検証可能性](../goal010/analysis/phase4_visibility/VISIBILITY_FEASIBILITY.json)の `NOT_VALIDATED_NO_VISIBILITY_CONCLUSION` を維持。silhouette/visible-face評価・可視率は計算しない |
| proxy | [旧1,350行の頑健性診断](../goal010/PROXY_FINDINGS_JA.md)を再利用。従来RQ2は `NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES` のまま |

O baselineは説明のdimensions_estimateだけ、P baselineは開示world_bboxだけから作った既存のfilled-box / six-face shellを使用する。新指標の結果に合わせた形状最適化はない。人工stone材質の5指標は適用外NULLとし、Builderの材質能力の対照には使わない。Pは正解bboxを含む情報対照であり、通常Oと同じ情報条件ではない。

## 指標の読み方

occupancy IoUは交差/和集合。新しいPrecisionは予測した占有位置がGTに属する割合、RecallはGT占有の回収率を区別する。余剰・欠落数、体積比も残す。F1は入力ごとにIoUの単調変換であり、独立の追加証拠には数えない。

surface F1@1/@2は6近傍boundary voxel中心間の距離に基づく。空洞内壁も含み、見える外表面や画像ピクセルを測る指標ではない。footprintはXZ占有投影で、間取りではない。column誤差、top-y誤差、6近傍連結性は階数・部屋・通行可能性を直接測らない。multi-scaleは数学的floorによる固定anchorのOR poolingで、穴や隙間を隠しうる。k8は補助診断として扱う。

absoluteはworld座標、alignedは元評価器のoccupancy最適shiftをそのまま使う。新指標ごとの位置合わせ、回転、scale調整は行わない。GTを使ったaligned値は実際の配置能力の証拠ではない。

材質はC=占有交差数、M=同位置かつexact ID一致数として、M/C、M/g、元material-aware IoU=M/(g+p−M)を併記する。M/Cは交差が少なくても高くなりうる。GT材質別macro recallは希少IDと頻出IDを等重みにするため、頻度を反映するM/gとは別の問いである。追加のalias・GT依存label修正は行わない。

## 全数の結果

60建築・960枠を保持し、934成功予測と480固定baselineの追加指標をすべて計算した。資源不足による未計算は0件、元のstrict値・全baseline IoU・新指標の独立検算は全件PASS。25既知モデル失敗（truncation19、validation6）と運用欠測1件は26行のNULLとして残した。[全件QA](analysis/3_full/metric_qa_report.json)。

以下はO/allのfilled-box対照、aligned、common-valid。各列はBuilder−boxの平均差 [95%探索的CI]。OpenAIは60建築/177 input pairs、Anthropicは60建築/179 pairsで、6指標すべて同じ分母である。

| 指標 | OpenAI Builder | Anthropic Builder |
| --- | --- | --- |
| occupancy IoU | −0.01196 [−0.02035, −0.00323] | −0.01161 [−0.01909, −0.00450] |
| occupancy Precision | +0.09257 [+0.06929, +0.11138] | +0.06528 [+0.04915, +0.08060] |
| occupancy Recall | −0.21123 [−0.23426, −0.18520] | −0.15356 [−0.17689, −0.12960] |
| surface F1@1 | +0.09929 [+0.08313, +0.11682] | +0.06452 [+0.04861, +0.08127] |
| surface F1@2 | +0.09199 [+0.07474, +0.11052] | +0.05513 [+0.03720, +0.07330] |
| XZ footprint IoU | +0.00581 [−0.00432, +0.01784] | +0.00019 [−0.00654, +0.00803] |

箱はGT占有をより多く覆う一方、予測占有に占める余剰の割合も高い。Builderは占有Recallが低い一方で、Precisionと近傍表面一致が高い。この対照は、同じスコアの高低だけでは生成の欠落と過剰を識別できなかった点を明確にする。footprintの差は小さくCIは0をまたぐ。既知失敗を0とする補助表では、IoU/P/R/surfaceの方向は維持したが、footprint点推定は−0.00245/−0.00076へ反転した（CIは両方0をまたぐ）。都合のよい欠測規約だけを選ばない。

O/P×filled/shell×2 Buildersの全48 core比較、provider別・absolute・補助failure0も含む全2,320比較を保存した。[全比較CSV](analysis/4_summary/metric_baselines_paired.csv)、[全core対照とanchor感度の所見](BASELINE_METRIC_FINDINGS_JA.md)。全2,320セルは2000/2000有効drawのCI付きで推定可能だった。1,560の順位診断は同じ適格ペア集合に揃えた記述で、新しい検定ではない。111,360の指標別候補ペアと延べ644の指標別除外はcanonical runに保持した。644は失敗した再建築数ではない。

旧80 IoU比較との照合では、N・両平均・全160 CI端点・補完数が完全一致した。差の平均は78/80で完全一致し、2件に最大1.39×10⁻¹⁷の浮動小数点丸め差があった。旧表は変更していない。

正常なempty successが1件（slot715、J/Anthropic）あり、これは25失敗に追加しない。aligned条件付き材質M/C等は933/934で定義可能、absolute M/Cは占有交差0の7件を除く927/934、absolute top-y誤差は共通footprintなし5件を除く929/934だった。[2,640分母表](analysis/4_summary/metric_denominators.csv)と[8群の信頼性表](analysis/4_summary/reliability_outcomes.csv)に、正常NULLと失敗・欠測を分けて残した。

EはOpenAI55/60成功・5 truncation、Anthropic46/60成功・14 truncation。占有IoU=1でも材質不一致がOpenAI22/36件、Anthropic4/14件残る。building034/OpenAIの一致材質は1794/4912でM/C=0.365228、M/gも同じだがmacro recallは4/6=0.666667となる。占有と材質転記、頻度と材質ID等重みは別の問いである。[材質・形状所見](MATERIAL_GEOMETRY_FINDINGS_JA.md)、[E容量診断図](analysis/4_summary/E_capacity_diagnostic.svg)。両Builderの成功集合が異なるため、これらの単群値だけで材質能力の勝敗を決めない。

比較差はBuilder−baseline。O/allでは建築内の適格説明差を平均し、その後に建築間を等重みにする。独立unitは最大60建築で、説明数や2 Buildersを独立Nへ水増ししない。95%探索的CIはplanned 60 buildingsを復元抽出するPCG64 seed20261017、2000 draws、type7 percentile。common-validを主表とし、既知の完了したモデル出力失敗だけに固定0を割り当てる補助表を分離する。未知欠測942、過去401、CPU資源不足、正常成功の定義上NULLを0にしない。新p値やHolm familyは作らない。

## 検証・保存

研究API生成、モデル照会、token-count API、キー検査、人手実験はすべて0件。旧予算USD21.56083625は歴史的な保守的予算計算額であり、請求書でも今回の追加課金でもない。

- テスト330件PASS（解析322＋コピー処理8）、fail/error/skipは0。開発途中の失敗JUnitも保存した。解析テストはCPU5.79秒・wall5.83秒・最大RSS113,748KiB。コピーfixtureの外部CPU/RSSは未計測で、解析値へ加算していない。
- source→1建築pilot→10建築pilot→全件→集計の5段階がPASS。合計CPU1,617.146秒、実時間1,617.774秒、最大RSS413,024KiB（約403.3MiB）。全件段階だけではCPU1,268.427秒。実装・対話・Git通信を含む総作業時間ではない。コピー処理CPU25.999秒は別計測。集計終了時の空き68,603,584,512 bytes。
- 最終整合監査もPASS。CPU25.268秒、実時間25.341秒、最大RSS1,082,060KiB（約1,056.7MiB）。大きな全ペアJSONも読んだこの監査が、今回計測したコマンド中の最大RSSだった。
- rootを維持し、実効設定はworkspace-write / auto_review。通常コマンドの成功を確認し、run書込み・Git操作には対象別の承認審査を使用した。OSレベル隔離は保証しない。新依存取得、ユーザー/SSH/OS変更は0。[実行環境](execution_environment.json)。
- 全件検証の最初の起動は、私が指定したcommit SHAの1文字欠落をguardが処理前に拒否した。成果物作成前に正しい固定SHAへ訂正し、コード・仕様は変えていない。サンドボックス起動エラーではない。[コマンドと実エラー](execution_preflight_recovery.json)。
- 29集計・receiptを[analysis/](analysis/)へコピーし、canonicalとの全decoded bytes一致を検証した。960行JSONだけは10MiBを超えるため、公開側を決定的gzipで保存し、解凍後のbytes/SHAを照合した。[CSV](analysis/3_full/new_metric_values_960.csv)、[JSON.gz](analysis/3_full/new_metric_values_960.json.gz)、[コピー証明](analysis/PUBLIC_COPY_RECEIPT.json)。元JSONと大きな全ペア表はcanonical run内に保全している。
- 原本・Goal007の20比較・Goal008の180/360・旧Goal010の結果とコードhashは不変。[最終整合監査](final_integrity_audit.json)、[ファイルmanifest](MANIFEST.json)。既存3 Draft/18 assetsもID・size・digest・draft・tag・targetが旧検証記録と一致し、ダウンロード・upload・Draft更新・publishは0。[読取り照合証拠](existing_drafts_readonly_verification.json)。

仕様と指標コードの固定commitは `d862a55134bba85a6a5e9677bcc6e94c719ca1aa`、コピー処理commitは `7ffd18744ef91b99fc9acb4e3e16fdb98d056ee1`。開始HEADは `080df2be3c5e949ff9bab6029d8b349fa7ed84d0`。保存先は既存専用branch `research/server-only-20261013`、main基準は `7309327efd1c19dda793b19e8d8e6ce0b5868db9`。最終commitとremote一致は文書を含む保存後に確認する。

canonical runは `/root/minecraft-runs/goal010-v2-20261016T233143Z`。既存runへ再実行せず、新しいUTC名でstage 0→1→2→3→4の順に実行する。各段階は前段receiptと固定commitの一致を必要とし、部分出力を上書きしない。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-server-v2-20261013/bin/python -B -m offline_evaluation.analysis.goal010v2.pipeline \
  --stage 0 --run-root /root/minecraft-runs/goal010-v2-YYYYMMDDTHHMMSSZ \
  --code-commit d862a55134bba85a6a5e9677bcc6e94c719ca1aa \
  --specification-commit d862a55134bba85a6a5e9677bcc6e94c719ca1aa
```

[要件対応表](REQUIREMENT_COVERAGE.json)、[材質・形状所見](MATERIAL_GEOMETRY_FINDINGS_JA.md)、[baseline指標所見](BASELINE_METRIC_FINDINGS_JA.md)、[研究判断](RESEARCH_DECISION_JA.md)に詳細をまとめる。
