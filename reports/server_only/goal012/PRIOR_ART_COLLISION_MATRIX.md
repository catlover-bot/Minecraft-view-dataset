**Goal012：一次文献との実質的な重複**

確認日2026-10-17。指定4論文と直接関連6論文について本文を確認した。網羅的systematic reviewではなく、未検出を不存在の証明に使わない。書誌・著者・DOI・版・確認節・アクセス制限は [SOURCES.json](SOURCES.json) のexternal_papersに保存した。arXiv本文を確認した論文について、出版版本文も同一と確認したとは言わない。外部実験は著者報告の読解で、再現実験ではない。

以下のHIGH/MEDIUM/LOWは広いアイデア／技術構成の重複度に対する本監査の判断で、論文の質や優劣ではない。Minecraftという対象の違いだけで重複をLOWにはしない。

| 一次資料 | 対象入力 → 中間表現 → 再構築出力 | GT・評価単位 | Builder条件 |
| --- | --- | --- | --- |
| [IGLU retrospective](https://proceedings.mlr.press/v220/kiseleva23a/kiseleva23a.pdf)（NeurIPS 2022 Competition Track／PMLR220） | Architect–Builder対話 → T5による座標・block type／subtask → 11×11×9の6色block環境 | 初期状態から目標状態への必要な追加・除去block。held-out subtaskに複数episode。 | baselineはfine-tuned T5-large＋決定的subtask化＋PPO実行。agentは自分のworldを観測する。 |
| [3D Building Generation in Minecraft via Large Language Models／T2BM](https://arxiv.org/html/2406.08751v1)（CoG2024） | user prompt → 詳細化prompt → JSON component IR＋repair → Minecraft建築 | promptの材質制約と構造要件。撮影元建物のvoxel GTではない。raw/refined設定を各50回評価。独立prompt数は未確認。 | GPT-3.5／GPT-4。構文・材質repairが出力を変更する。 |
| [RICO](https://aclanthology.org/2025.emnlp-main.1105.pdf)（EMNLP2025） | 画像 → 初期caption → 2D再生成画像 → caption修正（反復） | 元画像、QA／scene graph／object annotation、CAPTURE用reference captions。unitはimage-caption、編集比較は100画像。全benchmark総Nは未確認。 | FLUX.1-dev再生成、GPT-4o reviser、標準2 iteration。DPOによるRICO-Flashも扱う。 |
| [A Reconstruction-Based Framework for Caption Evaluation Beyond Reference Captions／CTTD](https://arxiv.org/html/2607.23235v1)（2026 preprint） | 画像・caption → 自然言語 → 2D画像 → 下流QA／grounding | task answer・box、元画像でのtask性能。CTTDは7k画像、約15質問／画像。質問数を独立画像Nとはしない。 | Qwen-Image、Qwen3-VL-8B judge、grounding用Qwen2.5-VL-72B。 |
| [Text2Shape](https://arxiv.org/html/1803.08495v1)（ACCV2018／出版2019） | 回転render観察から得た人間の自由記述 → joint embedding → colored voxel | ShapeNet chair/table CADのvoxelization。object split 80/10/10。厳密な集計unitの詳細は未確認。 | conditional Wasserstein GAN。Minecraftの離散材質IDではなくRGBを持つ占有grid。 |
| [DreamCraft](https://arxiv.org/html/2404.15538v1)（FDG2024） | 自由記述 → image-model score distillation／quantized NeRF → Minecraft blocks | conditioning caption。元建物のexact GT assetは収集していない。153 COCO／150 PMC名prompt、caption当たり5画像平均。 | 離散化、block分布・隣接制約。opaque solid blocksのみ。 |
| [Image2Text2Image](https://pure.uva.nl/ws/files/243811317/Image2Text2Image.pdf)（MMM2025、preprint2024） | 画像・caption → 自然言語 → 2D再生成画像 | 元画像、Flickr8kの人間判断、FOIL／M-HalDetect。unitはimage-caption／sentence pair。 | InstructBLIP/Vicuna-7B、Stable Diffusion 3 medium、DINOv2。 |
| [CAMScore: Evaluating Image Caption via Cycle-consistent Text-to-Image Generation](https://arxiv.org/html/2501.03567v2)（2025 preprint） | 画像・caption → 自然言語 → 2D画像・検出object・推定depth | 元画像とhuman caption判断。Flickr8k／Composite、Pascal-50Sのpair ranking。 | frozen FLUX.1-dev、CLIP ViT-B16、YOLOv8x、Depth Anything v2-Large。 |
| [What Do Single-view 3D Reconstruction Networks Learn?](https://arxiv.org/html/1905.03678v1)（CVPR2019） | 単一render画像 → latent／retrieval表現 → voxel／mesh | ShapeNet CAD、55 classes、shape split 70/10/20、5 views／shape。自然言語中間表現はない。 | OGN、AtlasNet、Matryoshka。object-centeredとviewer-centeredを比較。 |
| [Tanks and Temples](https://vladlen.info/papers/tanks-and-temples.pdf)（SIGGRAPH2017） | 実sceneのvideo/frame → SfM/MVS → point cloud | industrial laser scan。scene単位、GTへのregistrationとcropを伴う。自然言語はない。 | 複数のopen-source／商用SfM・MVS pipeline。 |

| 資料 | 評価指標・baseline | 今回との差・限界と元主張への競合 | 重複度 |
| --- | --- | --- | --- |
| IGLU | 必要変更数で重み付けたblock追加／除去F1。precision／recallも報告（集計重み詳細は未確認）。T5＋PPO baseline。 | 言語→grid→施工とGT評価は既出。本研究は保存画像説明・建築全体のexact材質・建築等重み診断。IGLUの変更F1はsurface F1@1ではなく、clarification課題も元RQ2ではない。 | **HIGH** |
| T2BM | completeness／材質制約satisfaction。raw対refined、GPT-3.5対GPT-4。 | JSON IR・repairは既出。satisfactionは人間満足度ではない。refinementは情報追加であり、同一情報prose/IR介入ではない。本研究O/Jにもその介入がない。 | **HIGH** |
| RICO | CapsBench、CompreCap、Amber、CAPTURE。初期caption、LaCLIP等、編集対照。 | 再構築を使うcaption**改善**は既出。生成器への依存がある。既知voxel GTの失敗診断とは異なるが、round-trip自体は新規にできない。 | **HIGH**（原理） |
| CTTD | QA等accuracy、grounding IoU、task平均。CaptionQA、CLIPImage、CLIPScore、FLEUR。 | 再構築型caption**評価**に直接競合。下流task成績を意味保持の代理信号とし、exact block一致とは違う。human品質相関未検証、generator/judge依存を本文が明記。本研究も同じ構成要素交絡を解決していない。 | **HIGH** |
| Text2Shape | occupancy IoU、color EMD、inception/class指標。GAN-INT-CLS、conditional GAN。 | 記述→voxelと形状／色の別評価は既出。一つの文に複数形状が適合する限界があり、単一GT不一致を意味的不適切さと同一視できない。本研究はfull buildingと厳密材質ID。 | **MEDIUM** |
| DreamCraft | CLIP R-precision。unconstrained NeRF＋後処理nearest-color量子化、ablations。 | text→Minecraftと見た目／材質意味の乖離は既出（App.C）。本研究のexact-GT材質対応上界・転記／容量分母が狭い相違。R-precisionは元RQ2の品質予測相関ではない。 | **HIGH** |
| Image2Text2Image | 再生成画像のembedding cosine、human判断とのKendall、hallucination判別。BLEU、METEOR、CIDEr、SPICE、CLIPScore等。 | image→text→reconstruction評価に直接先行。独立caption判断との検証があり、本研究のGT属性proxyで置き換えられない。 | **HIGH** |
| CAMScore | pixel／semantic／object／spatial scoreの学習融合、Kendall／ranking。従来caption metrics。 | 循環評価と複数空間診断は既出。推定depth・2D boxと既知voxel GTは異なる。検出category・非写実画像等の限界がある。本研究は新学習metricを提案していない。 | **HIGH**（原理） |
| Tatarchenko et al. | IoU、Chamfer、surface F-score/P/R。clustering、retrieval、GT-informed Oracle NN。 | 単純対照、frame依存、IoUとsurfaceによる順序変化、高precision／低recallは既出。10K mesh surface samplesと本研究のboundary voxel中心は同一実装ではないが、指標変更だけの新規性は弱い。 | **HIGH**（診断原理） |
| Knapitsch et al. | 距離閾値のP/R/F-scoreとpipeline間比較。 | 被覆と精度の分離・F-scoreは既出。laser point cloud・scene別閾値に対し本研究は整数voxel・1/2 blocks。離散化の違いだけで新指標とは呼べない。 | **MEDIUM**（指標） |

**本研究側の出典。** 同一情報O対照と反例は [Goal011 tradeoff](../goal011/TRADEOFF_EXPLANATION_JA.md)／[感度](../goal011/SENSITIVITY_REVIEW_JA.md)、geometry/material/capacityは [E診断](../goal011/E_INTERPRETATION_JA.md)、O/J/P/Eの意味は [Goal008表現監査](../goal008/audit/phase0_semantics.md)、proxyの位置付けは [Goal008定義](../goal008/definition.md)。これらと上の一次資料を比較した「今回との差」は本監査の推論であり、外部著者による本研究の評価ではない。対象や指標が違うため、論文間の数値ランキングは行わない。

**claim別の新規性判定。** text→3D、Minecraft Builder、JSON IR、round-trip caption評価、geometryと色／材質の区別、P/R/surfaceによる診断は既存構成要素である。残る候補は、固定画像説明と厳密block GTの対応、同一入力の寸法対照、全valid Eと打切りを分けた監査を組み合わせる限定的な実証設計である。同一設計が他にないという証明はなく、組合せだけで貢献が十分とも認定しない。順位と反証条件は [NOVELTY_AND_FALSIFIABILITY_JA.md](NOVELTY_AND_FALSIFIABILITY_JA.md)。

**アクセス上の留保。** IGLU landingは2022、PDFは2023と記載し、PDFにはlandingから欠けるMatthew Hoがあるため両方を保存した。Text2Shapeは会議年2018と出版年2019を分けた。T2BM／Text2Shape／DreamCraftの出版版本文同一性は未確認。RICO・CTTD・Image2Text2Image・CAMScoreは本文を読めたが、ローカルPDF hashは未取得。TatarchenkoはCVF取得失敗後にarXiv HTML本文を使用。Tanks and Templesはwebの容量制限後に著者PDFを認証なしで取得し、13頁をローカル抽出・hashした。現在のleaderboardは取得しておらず、現行順位を主張しない。
