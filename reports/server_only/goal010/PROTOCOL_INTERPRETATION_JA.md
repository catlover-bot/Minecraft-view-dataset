# Goal010 Phase0 プロトコル監査の解釈

POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC。本ノートは完了したPhase0の不変出力とPhase4の可視性検証可能性監査を読んだ解釈補足である。baseline・材質分析を再実行せず、原本の再評価、新たな研究統計、API送信は行っていない。

180件すべての保存requestが参照した画像は6枚であり、各建築の保存候補は12 viewであった。したがって記述生成の入力を「12画像」と報告する根拠はない。同じ建築について、3 provider間で6画像のbytesと順序、promptおよびsystem promptのhashが一致した。画像内容や提示順のprovider間差をこの保存データに帰す根拠はない。ただし、この一致だけでは記述生成の実験条件全体が統制されたとはいえない。監査件数は180 request、画像参照1080件、保存候補の一意画像720枚、独立評価建築60件である。720枚を全providerへの実送信枚数と解釈しない。

| 記述provider | 保存されたmodel ID | 記録max_tokens | request記録UTCの範囲（秒未満省略） |
| --- | --- | ---: | --- |
| OpenAI | gpt-5-mini | 1800 | 2026-02-28 15:03:32–15:29:34 |
| Anthropic | claude-haiku-4-5-20251001 | 1800 | 2026-03-01 04:49:59–05:04:35 |
| Gemini | gemini-3.1-pro-preview | 4096 | 2026-04-18 15:35:51–19:44:27 |

temperatureの記録値は3 providerとも0.2である。top_p、top_k、seedの未記録値を既定値の一致とみなさない。Geminiに保存されたllm_seed=-1も、provider間で共通の決定的seedを設定した証拠にはならない。モデルversion、出力上限、保存時期、単発の生成実現がproviderと結び付いているため、以降のprovider差はこれらを含む保存パイプラインの観測差として記述する。モデル固有能力への因果帰属や、出力上限だけを原因とする説明は支持されない。

created_atは保存記録の時刻である。Phase0のproducer監査では応答取得後にdescription/request記録を保存することが確認され、実API送信開始・終了時刻は記録されていない。上表は開始時刻、API処理時間、連続稼働期間を示さない。画像のscene metadata時刻とも区別する。

評価60建築は、元の100建築をGT occupied数・scene IDで順位付けし、5つの順位層（各20建築）内でseed=20261013の固定乱択を行った分割の再生と一致した。各層は評価12、smoke2、残り6で、合計60/10/30。保存された選択はmodel結果を使用していない。同一建築の3記述や2 Builder観測を独立な建築標本へ増やさない。また、60建築への条件付き結果からMinecraft建築一般への無条件の代表性を主張しない。

Oは保存された人間可読フィールド付きJSON、Jは同一typed JSON treeの再シリアライズである。同じ意味内容の束縛は確認されているが、O/Jはそれぞれ別の単発Builder生成なので、表記だけの決定的な因果効果とは扱わない。Pは開示GT属性、Eはexact paletteとinclusive cuboidを用いた完全離散仕様であり、Eを自然言語品質の母集団上限と呼ばない。

GTのstorage bboxとNPY配列の軸順YXZからworld XYZへの対応は、固定reader・GT descriptor/hashに束縛されている。storage paddingをoccupied bboxと同一視せず、撮影時generation origin、GTのworld座標原点、camera位置も別概念として扱う。strict translation評価は占有を最大化する単一shiftを全aligned指標へ共有する。材質集合F1の高さだけでは、各位置の材質一致を意味しない。

カメラに関してPhase0で確認できたのは、画像ファイルと記録metadataの対応である。planned/requested yaw・pitch・FOVと、記録されたactual XYZだけから実際の光学投影を復元できるとはいえない。actual yaw/pitch、eye offsetを含む光学中心、effective projection、画像と同期した深度・遮蔽対応の独立検証が不足している。保存FOVを実効投影の保証と書かず、metadataの存在から可視voxel割合、不可視部精度、内部構造推定性能を作らない。Phase4の検証可能性監査は2026-10-16 22:45:44 UTCに完了し、判定は `NOT_VALIDATED_NO_VISIBILITY_CONCLUSION` だった。保存720画像は960×540、記録FOVは70度、actual XYZは658 view、actual yaw/pitchは0件である。これらはmetadataの存在数であり、検証された可視性の割合ではない。可視/不可視voxel割合と6対12画像の可視差はnullのままで、数値的な可視性結論は出していない。[Phase4 receipt](analysis/PHASE4_RECEIPT.json)と[検証可能性出力](analysis/phase4_visibility/VISIBILITY_FEASIBILITY.json)がこの完了状態を束縛する。

Phase0 receiptは14個の読取りlock下で原本の前後一致を記録する。960固定slot、934 valid、25既知出力失敗、global942のHTTP520による欠測1件、範囲内961 attempts、歴史的991 callsを維持する。global940の元401と承認retryの選択関係を残し、成功を二重計上しない。unknown usageの2予約、既存checkpoint・GitHub検証証拠を保持し、USD21.56083625は請求額ではなく保守的予算値として扱う。今回の監査は既存GitHub証拠のローカル照合であり、新しいダウンロード検証ではない。Python接続禁止やlockの記録をOS隔離の保証と読み替えない。

出典は `/root/minecraft-runs/goal010-20261016T223832Z` のPhase0とPhase4の出力に限定した。Phase0 receiptに束縛された7ファイル（protocol manifest、2監査Markdown、SOURCE_BINDINGS、source gate、source inventory receipt、request audit JSON）と、Phase4 receiptが束縛する検証可能性JSON・限界Markdownを読取り、bytes/SHA一致を確認した。原本request/response本文や鍵への追加アクセス、可視性の結果計算はしていない。

| 証拠 | SHA-256 |
| --- | --- |
| PHASE0_RECEIPT.json | ba21f8db48030eb2410af20ce88dbc1a7e777d86ace7828237646955bf721bbe |
| phase0_protocol/PROTOCOL_MANIFEST.json | 42d5285b0d8ed03c5ba87d7f4cfd7232973e781e49881a15193f925bc5c7b6e2 |
| phase0_protocol/SOURCE_BINDINGS.json | cf6f1002328a2665cff12355bb4e7f3fee5ef105e820c504b49e870083caf3d7 |
| phase0_protocol/view_and_generation_request_audit.json | 34aa34534ded18be41a09a6b4e37f3194f361a5b9f4605eea373d59973b81987 |
| PHASE4_RECEIPT.json | a00e42af5c32529cbb4628298b1e860fd7dc7ae528c357f0ce1c58791d5910c5 |
| phase4_visibility/VISIBILITY_FEASIBILITY.json | d23cb693692c9677abd1afcd9884930f56edf88a33d748e5a3362c4c25c2ccc8 |

Phase0のコードcommitは `4a3b714ad139ee0f2836d756347b6bf02dbb9a8b`、事前固定したpost-hoc仕様commitは `87f85a2b6bc06a1ae30af8e5276d3aff67c58379`。本ノートの追加により既存Phase0出力は変更していない。
