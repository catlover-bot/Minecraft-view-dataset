# Goal010: pre-computation specification of post-hoc analyses

ラベルは **POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC**。従来の結果は既知である。
これは事前登録ではなく、新しい実データ解析の計算前に固定する後付け分析仕様である。
このファイルを専用ブランチへcommitした後にPhase0の詳細監査とCPU解析を行う。
仕様の変更が必要なら、理由・変更前後・既に閲覧した結果を別の追記に記録する。

## 0. 不変条件・実行順序

- 基点HEAD: `de3cb5e6d8bcec801219f376501c76ec0da926bc`。
- branch: `research/server-only-20261013`。main、既存Draft Releases、既存原本を変更しない。
- Goal007 MANIFEST SHA256: `ef05c2ed93247802e9a68a00d700d39a8934d6231c7787315ab37b01d2467714`。
- Goal008 MANIFEST SHA256: `20e4433b3a56132f6031cd4ac08da14566f69795ad412a47d174503b0c053df6`。
- 研究API生成・モデル照会・token-count API・キー検査・人手実験は全て0件。
- 既存Python環境を使用。新しい依存、OS変更、旧成果物の上書きはしない。
- ソースはverified MANIFESTから導出する。読取りlockとPython socket接続禁止の下で、各段階の前後にhashと選択provenanceを確認する。これはOSレベルの隔離保証ではない。
- 100建築の固定cohortを監査し、60評価・10 smoke・30 rest、独立unit最大60建築を維持する。
- 元の960 slots / 934 valid / 25 identified output failures / 1 operational missing、固定961 attemptsと歴史的991 calls、940の選択retryと元401、942の520欠測、unknown-usage予約2件を変更しない。
- USD21.56083625は過去の保守的予算計算値であり請求書ではない。今回の追加生成費用を計上しない。
- Goal007の20比較とGoal008の180 scores・360探索分析、コードhash・定義・p値を変更しない。
- Phase0 source/protocol監査 → Phase1 baseline → Phase2 provider contrasts → Phase3 E/material → Phase4 views → Phase5 proxy → synthesis。
- source不一致、実行中の有料生成、研究APIが必要になった場合は停止する。計算資源制限はmissingとして記録し近似で補完しない。可視性が未検証ならPhase5へ進む。

## 1. Phase0: source・protocol・撮影入力

180のO説明をscene/provider/input ID/source SHA/GT SHA/両Builderのselected slot/plan/attemptで束縛する。
全180 request JSONについて、実画像path/hash/count、候補meta.views、モデル名、prompt hash、temperature、top_p、seed、max_tokens、保存UTCを監査する。入力画像の実bytesとPNG寸法を調べる。欠損設定を推定で埋めない。
`description.request.json.created_at`は生成処理後のrequestレコード作成時刻であり、送信開始時刻と同一視しない。実装と保存証拠を照合する。
12候補viewsと実参照viewsを区別し、034の6枚・1800/4096設定を全体に先取りして一般化しない。
Oは人間可読フィールドを含む保存JSON、Jは同一typed treeの再シリアライズ、Pは開示GT属性、Eは固定実入力のschemaとする。
O/Jは別々の単発API生成であり、seed/temperature/top_pの明示固定がなければ純粋な表記変更の因果効果と呼ばない。
storage bbox、occupied bbox、world origin、撮影generation originを区別する。既存safe readerのNPY YXZ→XYZ変換を継承しpickleは使わない。
評価はserver_v2.2_strict_translation、occupancy最適の厳密平行移動、同じshiftで材質評価、rotation/scaleなし。失敗MAE/repairはnull。

成果物: `PROTOCOL_AUDIT.md`, `view_and_generation_request_audit.csv/json`, `SOURCE_BINDINGS.json`。
Phase0 PASSとこの仕様のcommit/hash束縛を後続関数の必須gateとする。

## 2. Phase1: 独立した幾何baseline

予測は480件: Oの180説明×2構造、Pの60建築×2構造。Builderごとに予測を複製しない。
構築関数はOなら`dimensions_estimate`だけ、Pなら`world_bbox`だけを受け取り、GT readerや材質・count・styleにはアクセスしない。全予測を保存してからGT評価を開始する。

### 入力・構造

- O: width→X、depth→Z、height→Y。正の有限JSON数値のみ（bool/string/nullを拒否）。整数化は既存proxy寸法変換と同じround-to-even、最小1。0/負値/欠損/非有限を拒否し、GT補完しない。正数の整数化は明記し、資源超過をclampや縮小で解決しない。暫定原点(0,0,0)。
- P: 開示world_bboxのinclusive整数xmin/xmax/ymin/ymax/zmin/zmaxのみ。voxel_count/material set/個別配置は使わない。原点はbbox最小座標。occupied bboxとstorage padded-air bboxの差は評価後に記録。
- filled-box: 全inclusiveセルを占有。six-face shell: 厚さ1の六面境界の和集合。幅1/2等の重複セルは集合化。材質は定数stoneであり材質スコアを論証に用いない。
- frozen cohort executor_limitsをそのまま使用。操作・座標・touch/live voxel・alignment/memory制限をfail-closedで適用。任意の入力別チューニングなし。
- 新しい予測actions/占有座標は新run rootだけに保存し、hash/bytesと失敗理由をmanifestへ記録。

### 評価・独立検算

主指標はstrict aligned occupancy IoU、補助はabsolute occupancy IoU。主要値は別実装でも検算する。
独立実装はGT占有格子の3次元累積和を使い、全整数平行移動の直方体交差数を厳密に数える。shellは外直方体から内直方体を引く。探索範囲は正の交差があり得る全移動、同点は既存規則の最小L1・XYZ辞書順。absolute値は座標集合の独立照合を使う。既存alignmentと交差数/shiftが一致することを要求する。独立法にも明示資源上限を適用し、未検算値をPASSにしない。
P filled-bboxのabsolute IoUはGTがbbox内であることを確認し`N_GT/bbox_volume`とも照合する。storage padded-airの差を記録する。

### ペア・集計

- 差は **Builder − baseline**。同じscene/provider/inputに対応するO、同じscene/P入力に対応するPを使用。
- 主解析common-valid。補助failure0は既知のモデル出力失敗だけをbounded IoU=0とする。unknown operational missing、baseline入力不適格/CPU失敗は0にせず除外。両方の分母・理由を全セル保存。
- 固定80集計: 条件O/P×2構造×2Builders×absolute/aligned×2policies=32、およびOの3provider別×2構造×2Builders×2metrics×2policies=48。
- Oの全provider集計は各建築内で適格description差を平均し、次に建築を等重み。Pは1建築1差。provider別も1建築1差。2Builderは別々に示しNを合算しない。
- planned frameは60建築。bootstrapはbuilding cluster、PCG64 seed20261017、2000 draws、95% type7 percentile CI。2以上の寄与建築かつ1900以上の有限drawを必要とする。欠測建築はplanned frameに残し、draw内の有効建築平均を用いる。N/pairs/finite draw数を記録。p値・有意性ラベルは出さない。
- per-input480行と全paired raw行、80集計を保存。GTサイズ別は元の5 size_rank_stratumと全体について、2条件×2構造×6層×2指標=48分布集計。建築内平均のmedian/Q1/Q3/min/maxと建築数・説明数を記録。欠測層も残す。

成果物: `baseline_predictions_manifest.json`, `baseline_per_building.csv`, `baseline_matched_deltas.csv`, `BASELINE_INTERPRETATION_JA.md`、補助paired/size表。

## 3. Phase2: description provider × Builder

全36 contrastsを固定: 2Builders×O/J×3 provider pairs×3指標。
方向はAnthropic−OpenAI、Gemini−OpenAI、Gemini−Anthropic。
指標はaligned occupancy IoU（主探索）、aligned dimension MAE、aligned material-aware IoU（副次）。MAEは小さい方が良い。
各contrastは同じscene/Builderのcommon-valid両descriptionだけ、1建築1差、等重み。
bootstrap frameはそのcontrastのcommon-valid建築（scene ID順）。PCG64 seed20261017、2000 draws、95% type7 CI、2建築以上・1900有限draw以上。baselineのplanned-frame欠測法との違いを報告。
全pairsと除外理由、平均差/CI/Nを保存。新p値や既存Holmの流用はしない。
モデルversion、実画像、prompt、max_tokens等を設定込みの交絡として記録し、providerの固有能力の因果効果とは呼ばない。

成果物: `description_builder_matched.csv`, `description_builder_confounds.json`, `FACTOR_INTERPRETATION_JA.md`。

## 4. Phase3: E control・材質・容量

実際のE schemaをverified manifestから確認する。予定する現行形式は`resmoke_v2_exact_cuboids_v1`、sorted paletteとinclusive 7-field cuboidsである。異なれば停止して仕様との差を報告する。
60建築すべてでGT→固定E bytes→reference fill actions→safe executorの座標・材質完全一致を要求。これはCPU controlでありLLM成功数に含めない。
120 outcomesをsuccess/truncation/validation/otherへ分け、valid Eを全数検算する。既存保存shiftをそのまま使い、同座標GT/predのexact material confusion・追加/削除/置換からabsolute/aligned元metricsを独立再計算する。指標別shiftの再選択は禁止。
占有ほぼ完全はIoU>=0.99、材質低値はmaterial-aware IoU<0.5という固定記述閾値。全件一覧と034 OpenAI Eを示す。操作数、palette外label、cuboid/operationの転記差は診断でありGT補修をしない。
capacityは実input/spec bytes・文字数・cuboid数・GT占有数、reference action量、保存usage、planで指定されたoutput上限を記録。token APIや未観測の完成出力推測は使わない。

### 事前固定の別計算lexical sensitivity

出典`tools/evaluate_rebuild_metrics.py`の`DIRECT_TYPE_NORM`、SHA256 `d67bd07007c9db56c532859763a41dbc26ced3d9c00e0905c24d38e6c5fb807c`。
次の狭い辞書だけをGT/predの両側へ決定的に適用し、同じabsolute/保存shiftで別列にmaterial IoU/set F1/matchesを計算する。記載外のlabelは不変。

```json
{"stone_bricks":"stonebrick","stone_brick":"stonebrick","stonebrick":"stonebrick","minecraft_stone_bricks":"stonebrick","minecraft_stone_brick":"stonebrick","brick_block":"brick","bricks":"brick"}
```

この歴史的語彙規則は物理的等価性を保証しない。stonebrickとbrickを同一化しない。slab/fence/planks/glass等の広い統合や、結果最大化による対応表探索はしない。元のstrict metric/入力/予測を変更しない。

成果物: `E_material_confusion.csv`, `E_error_taxonomy.json`, `E_capacity_diagnostics.csv`, `MATERIAL_VALIDITY_JA.md`、control/strict検算/alias別表。

## 5. Phase4: visibility feasibility

Phase0で凍結した実画像/hash、候補12viewsと実参照数、planned yaw/pitch、camera XYZ、FOV、画像寸法、UTC記録を整理する。
保存FOVがrequested hintであれば実投影と同一視しない。actual yaw/pitch、眼のoffset、投影規約、遮蔽再現を実RGBとの独立照合で検証できない場合、`NOT_VALIDATED_NO_VISIBILITY_CONCLUSION`とする。
その場合可視割合/不可視割合/領域別IoUは計算せず、実画像coverage表と不足理由を保存してPhase5へ進む。生成画像を観測画像の代わりにしない。可視面と内部占有volumeを混同しない。

成果物: `image_request_view_audit.csv`, `VISIBILITY_FEASIBILITY.json`, `VISIBILITY_LIMITATIONS_JA.md`。検証不能なら`visibility_region_metrics.csv`は作らず理由を明記。

## 6. Phase5: proxy robustness（固定1350行）

元の180 proxyとO slot結合を保持。元RQ2は`NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES`のまま。
9features: Goal008の凍結5 featuresと、compositeの4componentを各1つ除外した4算術感度値。残りの固定重み0.30/0.25/0.25/0.20を和1に再正規化する。残存component欠損ならnull。新しい品質尺度・学習モデル・独立因果寄与とは呼ばない。
観測共変量5個: 元O UTF-8 decoded文字数、固定HUMAN_FIELDS内の再帰的string値文字数（dict keys/numbers/strip後空文字を除く）、GT occupied count、inclusive storage bbox volume、count/volume密度。
18 scopes: 全体1、description provider3、元size quintile5、O text length tertile3、storage volume tertile3、density tertile3。length cutpointsはplanned180、GT cutpointsはunique60。type7の1/3,2/3分位、low<=q1/middle(q1,q2]/high>q2。同値を分割せず、重複cutpoint/空scopeを明示する。
3targets: OpenAI Builder、Anthropic Builder、両Builderの同description平均。2policies: valid、既知output failureだけIoU=0。unknown missingは補完せず、両Builder平均は両者がpolicy上適格な場合だけ。

- 9features×18scopes×3targets×2policies = 972 pooled Spearman。
- 全scopeに対する9features×3targets×2policies×2methods = 108（building内centered Pearson、およびbuilding内midrankをcenteredした相関）。
- 5covariates×9scopes（全体/provider/元size層）×3targets×2policies = 270 pooled Spearman confound-outcome。
- 合計1350行を定数/不足/nullも含め保存。correlationは適格description pair等重み、Nはbuilding数とpair数を併記する。
- bootstrap frameは各固定scopeのplanned buildings。cluster再標本化で同一建築の全descriptionと両Builder結合を保つ。pooled ranksは複製後の観測全体で再計算。within法はcomplete pairのbuilding内でcenter/rankし、その後buildingを再標本化する。
- PCG64 seed20261017、2000 draws、95% type7 percentile CI、2寄与建築以上・1900有限draw以上。定数/不足ではpointまたはCIをnullとしstatusを残す。p値・Holm・CV・fitは追加しない。

成果物: `proxy_robustness.csv`, `PROXY_VALIDITY_JA.md`、全設定/分母/層境界のJSON。

## 7. QA・保存・判定

合成テストは漏洩禁止、P限定bbox、negative/inclusive/degenerate shell、重複、YXZ→XYZ、厳密材質とalias不変、E復号control、既知/unknown失敗、cluster bootstrap、providerペア、view/meta/settings/UTC、Python network blockade、safe paths/hash/asset bindingを覆う。
主要baseline値の独立実装一致と全valid E旧値一致、Phase前後source invarianceを必須にする。
新code・小さいaggregate表/日本語報告だけをGitへallowlist保存。raw responses/巨大voxel/認証情報/private URLsは保存しない。単一新規Gitファイル10MiB以下。大きな予測は新run rootに保持。既存Draftは更新しない。
最終READMEとRESEARCH_DECISION_JAは、baselineに負ける場合も含め研究主張の必要な修正を示す。観測交絡、公開GT情報の強いP/E control、弱いO箱baseline、未検証可視性を区別する。追加の最小独立実験は提案だけで実行しない。
新旧source hash、tests、CPU/elapsed/RSS/disk、API0、commit/remote/main/cleanを実測で報告。途中開始・部分完了を全完了としない。
