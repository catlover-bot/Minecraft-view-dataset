Goal014はGoal013Aの結果を知った後の **POST_HOC_EXPLORATORY** である。事前登録、独立再現実験、人間品質尺度の検証ではない。この仕様を含むcommitを実計算前に固定し、実行CLIの `--plan-commit` と実行receiptへ束縛する。

対象は元60建築×3 providersの180予定summary、非空178。Gemini building_054/061は空のままNULL。原scoreは `MEAN6_CLIPSCORE_VITB32_CPU_FP32_SUMMARY_ONLY_GOAL013A`。元175/177/174 paired rowsの関連結果、元20比較、960 slots/934 success/25出力失敗/HTTP520欠測1、991 historical calls、不明usage予約2を変更しない。元scoreと元CIは読み取り専用である。

唯一のモデルは公式OpenAI CLIP ViT-B/32、checkpoint SHA256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`。Goal013Aの固定コード・config・依存・tokenizer・BPE・前処理を継承する。CPU float32、eval/inference_mode、prefix `A photo depicts `、truncate=True、context77、SOT/EOTを含むpayload75、prefix3、summary最大72token。元224 bicubic resize→center crop→RGB→tensor→normalizeをそのまま使う。元960×540画像の残存面積56.28%で、全建築がcrop内に残るとは推測しない。数値許容差はatol=rtol=1e-5、token/hash/順序は完全一致。許容差をtie閾値に転用しない。

Q-A：全178 text×60候補scene×6視点=64,080 feature対の算術を行う。候補sceneは元60 IDの辞書順、各sceneの6枚はsource bindingの順序。GT/Builder/proxyによる選別はしない。同一summaryに対し全59他建築を使い、来歴上非対応の画像対照と呼ぶ。内容が必ず偽になる負例とは呼ばない。

各viewはL2正規化特徴の内積に `2.5*max(dot,0)`、候補得点s_ijはその6値の算術平均。特徴平均後cosine、最良view選択、0–1clip、logit scaleは使わない。元対応1068 view値と178平均を先に照合し、token/model/source/transform/hash不一致または許容差超過なら停止する。元scoreを再計算値で置換しない。

唯一の主診断は `D_i=s_i,b(i)-mean(s_ij,j!=b(i))`。補助は `W_i=(strict_wins+0.5*exact_ties)/59`。他候補平均・population SD、D/W、tie数を180予定行に保存。欠測2行はNULLと理由を保持。原Goal013Aのconcordanceのtie除外とは異なるhalf-tie規則である。建築内利用可能summary平均を計算し、その後建築を等重みで平均する。D=0/W=0.5は交換可能な無情報モデルの参照値で、事後的PASS閾値・p値ではない。

永続特徴は存在と真正性を確認した場合のみ再利用する。元runに特徴artifactがなければ360 unique画像と178非空summaryを同一adapterで一巡だけencodeし、新run/privateへfloat32数値npy（allow_pickle=False）と順序/hash metadataを保存。原著者parityはscene順先頭の最初のviewと最初の非空summaryの1対だけを追加確認する。parityの反復encode件数も別計数する。特徴・原文・token列・元画像・重みはGitへ新規公開しない。178×60行列もprivateに保管する。

Q-Aを凍結した後にのみ保存outcomeを解析用に結合する。モデルへ渡すのは原summaryと6画像のみで、ID/provider/outcomeはtextに含めない。

Q-B：Goal010v2 MANIFEST→canonical stage receipt→保存済みaligned surface_f1_at_1を解決する。同じscene/provider/Builder/O input ID/hash/selected attempt/pred・GT・元occupancy shift hashを検証する。新しい指標・alignment・surface最適化は計算しない。surfaceは6近傍boundary voxel中心の距離で空洞内壁を含み、画像可視外観・mesh・意味的間取りとは呼ばない。

Builder2種を別々に扱い、固定score・O aligned occupancy IoU・surface F1@1が全てfiniteかつ元validの同じcommon-valid集合を各Builderで一度固定する。その集合だけでrho_IoU、rho_surface、delta_rho=rho_surface-rho_IoUの計6項目。元Goal013Aと集合が同じか、行ID/mask・N・値を保存する。deltaは同じdrawの2係数の差を取る。both-Builder平均、新failure0表、他指標は追加しない。既定failure0は旧報告の参照だけとする。

Q-C：元truncated=false119件/true59件の2群のみ。元scoreと元valid O IoUの共通集合から各群×2Builderの4係数とCI。群をprimaryへ昇格しない。群ごとprovider件数と元summary token長min/median/maxだけ記述する。追加のprovider相関・length相関/補正・回帰・閾値探索はしない。群分けはprovider/文長/内容/難度の選択を含むため、群差は切捨ての因果効果ではない。

全12項目は同一のplanned60建築辞書順frameからNumPy Generator(PCG64)、seed20261017、2000×60復元抽出index行列を共有する。欠測建築をframeから除かず、建築多重度を保持する。Q-Aはquery建築平均の抽出多重度付き平均で候補galleryは固定。このCIは固定galleryに条件付けたquery建築の不確実性で、新規gallery/cohortへのCIではない。Q-B/Cは各drawでprovider行の複製を保ったpooled Spearmanを再順位化する。完全一致tieに平均順位を用いる。Q-B/C点推定は記述行等重みで、建築等重み係数とは呼ばない。

元点推定が定義され、寄与建築2以上、finite draw1900以上の場合だけfinite drawのtype7 linear 2.5/97.5 percentile CIを報告する。定数vector・不足・非有限・未定義はNULLと理由。不定drawを補充しない。全推定の予定/有効行、寄与建築、mask/行ID、finite/undefined draw数を保存する。p値、significance、新Holm、equivalence検定は0。2Builderを独立な追試とは数えない。

資源上限はGoal全体CPU1200秒/wall1800秒。開始UTCはENTRY_STATEの時刻を使用し、準備・test・保全・解析の計測値を累積する。RSS目標4GiB、上限min(6GiB,利用可能量から安全余裕2GiBを引いた値)。batch≤8/workers0/thread1。新run1GiB以下、空き10GiB以上。上限に達したら保存済み結果を保全し停止、値や対象を増やさない。開始/終了の全体保全hashは各1回を基本とする。段階ごとは必要な固定入力hashだけ照合する。監査guardはOS隔離の保証ではない。

新取得/install/外部研究API/モデルAPI/キー照会は0。既存venv/cache/OS/SSH/Codex設定を書き換えず、TMPDIR/XDG/TORCH/PIP cacheとbytecodeを新runへ限定または無効化する。ネットワークは専用branchへのGit保存のみ。既存Draft操作、再建築、論文、次Goalは作成しない。

新fixtureはsource/feature順序・score不一致、per-view clip後mean6、非有限/zero norm/空summary、59候補から自身除外、候補並替え不変、exact ties、bootstrap多重度/欠測frame/定数/paired差、Q-C flag/分母、旧path書込み拒否を検証する。旧全テストは実行しない。

完了時は実装parity、固定gallery識別信号、Minecraft人間品質妥当性未検証、RQ2限定版、原RQ2未検証を区別する。どの結果でも元Goal013Aの推定を変更せず、診断が識別したことと未識別なことを日本語RESEARCH_DECISION_JA/READMEにまとめ、この12項目で終了する。
