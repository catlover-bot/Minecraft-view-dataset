**Goal013A：実採点前に固定する唯一の採点方法**

`POST_HOC_EXPLORATORY / GT_INDEPENDENT_SUMMARY_ALIGNMENT`。これは過去の研究結果を知った後の計算前固定であり、preregistrationではない。Goal013Aの新しい明示承認で専用領域への公開ソフト・重み取得とCPU推論を許可する。旧Goal013の採点0・採点器不在という終了記録は変更しない。

本仕様の固定時点では実モデル推論未実行。実測hashと専用環境を `SCORER_CONFIG.json`・`EXECUTION_SPEC.md`・取得receiptへ統合した。コードと別の仕様commitを検証してから実モデルを動かす。期待値と実測値を別に記録している。

| 固定項目 | 値 |
| --- | --- |
| score ID | `MEAN6_CLIPSCORE_VITB32_CPU_FP32_SUMMARY_ONLY_GOAL013A` |
| 著者実装 | `jmhessel/clipscore@1036465276513621f77f1c2208d742e4a430781f` |
| CLIP実装 | `openai/CLIP@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` |
| backbone | OpenAI CLIP ViT-B/32、224入力、context 77 |
| expected checkpoint SHA256 | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |
| device / dtype | CPU / float32、`jit=False`、`eval()`、adapterは`inference_mode()` |
| prefix | `A photo depicts `（末尾空白を含む） |
| batch / workers / torch threads | batch 1–8、DataLoader workers 0、intra/inter-op各1 |
| source / dependency / checkpoint実測hash | 実行前の取得manifestとroot仕様に必須。nullや期待値だけならロード禁止 |

論文はHessel et al. (2021), [CLIPScore: A Reference-free Evaluation Metric for Image Captioning](https://aclanthology.org/2021.emnlp-main.595/)、DOI `10.18653/v1/2021.emnlp-main.595`。採点式とCPU経路は[固定著者ソース](https://raw.githubusercontent.com/jmhessel/clipscore/1036465276513621f77f1c2208d742e4a430781f/clipscore.py)、tokenizationとloaderは[固定CLIPソース](https://raw.githubusercontent.com/openai/CLIP/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/clip.py)に対応する。各ソースのMIT LICENSEは専用sourcesに保存し、実測hashで束縛する。著者ソースや重み本体はGitへ複製しない。

**計算と対象**

原文JSONからdecodeした`summary`だけにprefixを付ける。画像・textの特徴を別々にL2正規化し、各画像対を `q = 2.5 × max(dot(v,t),0)` とする。NumPy 1.21以降の著者経路に従い、float32の二乗和・平方根・除算・積和を用いる。1への上限clip、百分率化、softmax、logit scaleを加えない。

6個の有限qの算術平均をprimaryにし、同じ6値のmin、max、population SD（ddof=0）を保存する。集約は保存されたfloat32 qをPython floatへ変換し、`math.fsum`で計算する。これは本研究のmean6集約であり、検証済みmulti-view品質尺度ではない。1視点でも欠測・非有限なら部分平均を作らずscoreをnullにする。zero norm、誤ったdtype・次元は計算エラー。空summaryを0点へ補完しない。

原本180行と1080画像参照を保持する。非空178行なら実採点可能対は最大1068で、Gemini/building_054・061の空summaryは欠測のまま。scorerへ渡すのはsummaryと6画像だけで、scene/provider ID、GT、bbox、Builder結果、proxyを入力しない。sourceとmodelのSHA不一致はrun停止条件である。

**tokenの実列と代表性**

同一の固定BPE tokenizerでprefix列P、summary列S、結合列Tを取得する。context 77にはSOT/EOTが含まれ、payload上限は75。実際の`truncate=True`の77位置を独立に再構成した列と照合し、実列・hash・special/padding mask・モデルのargmax pooling位置を記録する。token ID 0はpayloadにも現れ得るため、nonzero ID数を有効token位置数と同一視しない。

`T=P||S`を確認でき、予約SOT/EOTがpayloadに混入せず、cleanup後Sが非空の場合だけcoverageを`VERIFIED`にする。その場合 `retained=min(len(S),max(0,75-len(P)))`、`dropped=len(S)-retained`、未読率`dropped/len(S)`とする。summary原文、prefix付き文字列、P/S/T、実77 IDsへ別々のSHAを持たせる。予約文字列は大文字小文字やtokenizer cleanupを経たID混入も検出する。未確認coverageを文字数から補わず、Phase 2へ入れない。

全体またはいずれかのdescription providerで、非空summaryの過半数の未読率が0.5を超えれば、`MEASURE_NOT_REPRESENTATIVE_OF_SUMMARY`として関連分析を止める。正確に半数、正確に0.5はこの超過条件ではない。未確認coverageは分母から消さない。既知の超過件数だけでは過半数でなくても、未確認全件が超過なら過半数になり得るgroupは`INDETERMINATE_TOKEN_COVERAGE`とし、合格を推測しない。非空0のgroupも未判定とする。全groupのPASS後でも未確認行は除外し、Minecraft領域の人間品質妥当性は`NOT_VALIDATED`のままである。

**前処理とcrop**

著者のResize短辺224・bicubic→CenterCrop224→RGB→ToTensor→所定mean/std Normalizeを維持する。PILのbicubic処理を使用し、EXIF転置や事前paddingを追加しない。元width/height、実resize width/height、半開区間のcrop矩形（resize座標と原画像座標）、残る面積率を保存する。cropが建物全体を含むかは未検証とする。

参考cropはGoal013のSHA `b52ba0e3280e23661710cb9c572ec2b298c40d28a54181ac247e40840f674107` のbindingをscene ID順に並べた先頭3建築、**building_001・003・005の第1要求画像（meta index 0）**だけに固定する。得点に基づく選別はしない。crop画像は新canonical runへ保存し、Gitへ登録しない。

**著者関数とのparity**

実測SHAを検証した`clipscore.py`から、`CLIPCapDataset`、`CLIPImageDataset`、`extract_all_captions`、`extract_all_images`、`get_clip_score`の元ASTを抽出して実行する。関数本文を再実装したものをreferenceと呼ばない。source SHA、元行番号、各AST SHA、除外したimportを記録する。無関係な参照caption評価・CLIは実行しない。NumPy>=1.21経路なのでsklearnは呼ばない。batch<=8/workers=0を渡すwrapperと進捗表示の無効化だけを変更として明示する。

| 比較 | 固定許容差 |
| --- | --- |
| token IDs | 完全一致 |
| 前処理tensor | atol=0、rtol=0 |
| L2正規化特徴 | atol=1e-5、rtol=1e-5 |
| per-image q／独立float64 scalar検算 | atol=1e-5、rtol=1e-5 |
| cacheから同一特徴を再取得 | 完全一致 |

固定参照入力の選択・source hashesはroot実行仕様に束縛する。元`get_clip_score`を実行し、token・tensor・特徴・q、batch1対batch<=8、cache有無、replay、6値の順序変更を確認する。合成fixtureの成功は実モデルparity PASSではない。2021年GPU/float16値とのbit一致を要求・主張しない。実測差が上限を超えたら停止し、許容差を結果に合わせて広げない。

**cache、取得との分離、後続解析**

画像cacheは画像bytes SHA・固定transform SHA・weight SHA、text cacheはprefix付き原文SHA・tokenizer/BPE identity・weight SHAをkeyとする。同一固定session内だけで再利用し、返却配列はcopyとする。checkpointはlocal絶対pathを全bytes hash後にロードし、download関数は推論時に拒否する。torch deterministic algorithmsを有効にするが、他の環境への完全決定性は保証しない。

CPU pilot、資源上限、専用出力先、依存実測hashとcommit gateはroot runnerの実行仕様へ束縛する。scoreを確定してからのみGoal013の固定Phase 2定義を用いる。`COMPUTATION_VALID`、token代表性gate、関連分析の実分母、人間品質妥当性を別々に報告し、source PASSや採点成功を旧RQ2の検証完了へ読み替えない。
