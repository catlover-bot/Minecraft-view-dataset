# Exact Oracleの材質・転記・出力容量：観測結果

POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC。60建築すべてでGT→固定E入力→参照fill操作→CPU実行の完全一致を確認した。一方、実際のLLM出力120件のうち19件はtruncation、101 valid件にも「占有位置はほぼ正しいが材質対応は低い」例が8件あった。Eを自然言語理解の理想上限、占有IoUを材質再現の代用として扱う主張には修正が必要である。

## 完了確認と出典

Phase3はE_AUDIT_PASS。Phase0〜3のPASS receiptと全束縛成果物、baseline NPY参照を読取り検証した。原本の前後hash一致、全101 validの保存指標との一致が記録されている。この報告は完成済み出力の記述集計であり、追加の統計検定・相関fit・alignment search・API生成を実施していない。canonicalのMATERIAL_VALIDITY_JA.mdと元のstrict metricsは変更していない。

- 実行code commit: `4a3b714ad139ee0f2836d756347b6bf02dbb9a8b`。
- 固定仕様SHA256: `e52ead81c32c7b76b3f93d226af6a715d39183c14437d21c23c49b9e7f221943`。
- Phase3 receipt SHA256: `9c7cfdc748cad599cee8f1e4cfa6c56a3da06213bbb401eb6a4f113236c3f037`。
- E_AUDIT_MANIFEST SHA256: `e31876cffe2b4f2100f5d25e452fdcaad11eac9c216ad109f1e0127ccbfd31fc`。
- E_error_taxonomy SHA256: `813d2e85600cf7e14111c233d49c3a0357f31d622837d406f1f107ba82be522d`。
- 出典: [全120結果と診断](analysis/phase3_exact_oracle/E_error_taxonomy.json)、[材質混同行列](analysis/phase3_exact_oracle/E_material_confusion.csv)、[容量診断](analysis/phase3_exact_oracle/E_capacity_diagnostics.csv)、[60 CPU controls](analysis/phase3_exact_oracle/E_reference_controls.json)、[元の監査報告](analysis/phase3_exact_oracle/MATERIAL_VALIDITY_JA.md)。

## 60件のCPU controlと実生成の区別

実入力はresmoke_v2_exact_cuboids_v1で、sorted exact paletteとinclusiveな7フィールドcuboidsを使用する。古いZ-run形式を実入力と取り違えていない。全60建築で保存GTの座標・厳密材質から作成したE bytesが固定入力と一致し、復号と参照操作列の安全実行も同じGT grid hashを再現した。storage bbox・negative world origin・NPY YXZ→XYZ変換も固定された入力規約で扱う。

このcontrolは固定入力・decoder・executorの整合を確認する。60件をLLM成功数へ加えず、LLMが同じ操作列を指定上限内で生成できる保証や、参照列が最短である証明にはしない。

| Builder | 予定E | valid success | truncation | validation | other |
|---|---:|---:|---:|---:|---:|
| OpenAI | 60 | 55 | 5 | 0 | 0 |
| Anthropic | 60 | 46 | 14 | 0 | 0 |

validは構文・検証・実行を通過したという状態で、GTとの完全一致を意味しない。全101 validで保存parsed→voxelをCPU再実行して同一性を確認し、元の保存shiftを固定したabsolute/alignedの全指標を独立した座標×材質confusionから再計算した。占有最適shiftの再選択、材質に有利な別shift、rotation・scaleは行っていない。19 truncationの未知の完成形を捏造せず、MAE/repairを0へ置換していない。

## valid条件付き平均と全8例

| Builder | valid建築N | aligned占有IoU平均 | aligned材質対応IoU平均 | 材質集合F1平均 |
|---|---:|---:|---:|---:|
| OpenAI | 55 | 0.9218936 | 0.8273725 | 0.9617334 |
| Anthropic | 46 | 0.9918719 | 0.8073928 | 0.9737827 |

55件と46件という異なる成功集合に条件づけた平均であり、この並びからBuilder固有能力の因果差や総合順位を結論しない。失敗率、材質対応、形状を分けて読む。材質集合F1が高くても各座標への割当が正しいとは限らない。

計算前に固定した「aligned占有IoU ≥ 0.99、材質対応IoU < 0.5」に該当する全8例を示す。これは説明用の固定閾値で、母集団の診断閾値や因果分類ではない。

| 建築 | Builder | 占有IoU | 材質対応IoU | 材質集合F1 | 置換voxel数 | GT占有数 |
|---|---|---:|---:|---:|---:|---:|
| 016 | Anthropic | 0.995499 | 0.280488 | 0.909091 | 2233 | 3999 |
| 026 | OpenAI | 1.000000 | 0.264392 | 0.933333 | 2447 | 4206 |
| 034 | OpenAI | 1.000000 | 0.223412 | 0.909091 | 3118 | 4912 |
| 047 | Anthropic | 0.991091 | 0.203410 | 1.000000 | 2645 | 4021 |
| 059 | Anthropic | 0.999816 | 0.427388 | 0.857143 | 2176 | 5426 |
| 064 | Anthropic | 0.999651 | 0.471397 | 1.000000 | 2055 | 5724 |
| 065 | Anthropic | 1.000000 | 0.121442 | 0.923077 | 3534 | 4511 |
| 095 | Anthropic | 0.990046 | 0.410347 | 0.923077 | 991 | 2402 |

047と064では材質集合F1が1でも、それぞれ2645、2055 voxelsの材質置換が残る。材質の種類が揃うことと空間上の材質割当の一致を区別する必要がある。

## building034：位置の完全一致と材質の転記違い

OpenAI E（global526）はGT・予測とも4912 occupied voxels、占有intersection/unionとも4912、占有IoU=1、寸法MAE=0である。保存shiftは[0,0,0]でabsolute/alignedの値も同じ。一方、厳密材質一致は1794、材質置換は3118、追加・削除は各0。

材質対応IoUは座標×材質の集合IoUであるため、1794 / (4912 + 4912 − 1794) = 1794 / 8030 = 0.2234122042となる。材質集合F1は0.9090909で、この値だけなら大きな配置誤りを見落とす。置換・total normalized repairは3118/4912=0.634772で、分母はGT occupied countである。

保存された混同行列のaligned行は以下の全6行。absolute行も同じで、二重に加算しない。

| GT材質 | 予測材質 | voxel数 | 種別 |
|---|---|---:|---|
| brick_block | brick_block | 568 | 一致 |
| glass | brick_block | 160 | 厳密ID置換 |
| quartz_block | quartz_block | 1033 | 一致 |
| sandstone | glass | 2958 | 厳密ID置換 |
| sea_lantern | sea_lantern | 3 | 一致 |
| stone_slab | stone_slab | 190 | 一致 |

実際の置換はglass→brick_blockの160 voxels、sandstone→glassの2958 voxelsで、合計3118になる。出力材質はすべてE palette内だったので、palette外IDの検出だけではこの誤りを捉えられない。

参照cuboidsは428、出力fill操作も428。全428境界が存在したが、正しい材質labelを持つ境界は202だけだった。86操作でglassがbrick_blockへ、140操作でsandstoneがglassへ置換されていた。境界の転記数とvoxel置換数は単位が異なる（86/140操作に対し160/2958 voxels）。別の正しい分割や上書き順序も可能なので、一般には境界不一致だけを欠落と判定せず、最終voxel confusionを優先する。

同じ034のAnthropic E（global527）は428操作・428境界の材質が一致し、4912 voxelsの厳密一致、占有IoUと材質対応IoUがともに1だった。これは同じ固定入力に対する一つの対照例であり、単一例をBuilderの一般的な優劣や因果効果に広げない。

## 狭いlexical alias感度：改善したvalid例は0/101

計算前に固定した歴史的DIRECT_TYPE_NORMの狭い辞書だけをGTと予測の両側へ適用した。stone_bricks/stone_brick/minecraft_stone_bricks/minecraft_stone_brick→stonebrick、brick_block/bricks→brickで、その他は不変である。stonebrickとbrickは統合せず、slab/fence/planks/glassの広い統合もしていない。

全101 validのabsolute/alignedで、材質対応IoU・材質一致数・材質集合F1が変わった例は0件だった。034の0.223412も不変であり、この規則でのalias救済は観測されない。だからといって全てのMinecraft名の物理的同値を検証したとはいえない。結果に合わせた辞書拡張・GT-fit・strict値の置換は行っていない。

## 容量：カテゴリ別の保存観測

以下は保存されたcapacity_group_summaryの中央値である。validation/otherは両BuilderでN=0のため中央値null。説明変数と失敗の相関・回帰・新しい検定は実施していない。

| Builder | 結果 | N | 入力bytes中央値 | 入力tokens中央値 | cuboids中央値 | GT占有中央値 | 出力tokens中央値 | 参照操作JSON bytes中央値 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| OpenAI | success | 55 | 12466 | 8891 | 554 | 4795 | 19925 | 44664 |
| OpenAI | truncation | 5 | 19444 | 13741 | 897 | 6667 | 32768 | 70907 |
| Anthropic | success | 46 | 11338.5 | 8770.5 | 494 | 4609.5 | 22841 | 39807.5 |
| Anthropic | truncation | 14 | 19347.5 | 14435.5 | 896 | 6018 | 32768 | 71194 |

全120件のoutput_token_capは各元planから読み取った32768だった。OpenAIの5 truncation、Anthropicの14 truncationはいずれも保存出力usageが32768で上限比1.0。validの出力token中央値は19925（上限比0.608063）と22841（0.697052）で、最大は29718と32393である。上限到達という観測は確認できるが、上限を増やせば正しい完成形が出る、長さだけが失敗原因である、という反事実は識別しない。

OpenAIのreasoning tokensはvalid中央値512、truncation中央値448、truncation最大12608だった。Anthropicではこの欄が未報告で全件nullであり、0と見なさない。provider間でtokenの分解・計数規約が同じだとも仮定しない。

034では固定入力9944 UTF-8 bytes（spec9276 bytes）、428 cuboids、同材質Z-run参考数1520、参照操作列34536 bytes。OpenAIの保存input/output tokensは7109/17371（reasoning512）、Anthropicは7811/17518で、両者とも自然終了していた。この例の材質誤割当はtruncation例ではない。

参照操作列はcuboidごとに1 fillという一つの正解構成で、最小操作数・最小token列の証明ではない。参照token数はnull、NOT_CERTIFIED_NO_LOCAL_TOKENIZATIONのままであり、bytesをtokensへ換算しない。ローカルexecutor上限は元cohortからの50000 operations、2000000 touched visits、1000000 live voxels、8000000 JSON bytes等を保存し、実行controlとLLM出力予算を混同しない。

## 研究主張への含意

固定E仕様は全60建築でCPUから完全に再構築できた。しかしLLM経由では出力打切りと、座標を保った材質labelの誤割当が残る。よってEは「完全な離散仕様から操作を転記・実行する条件」の診断であり、自然言語の理解力や情報損失だけを分離した上限条件ではない。

占有と材質を別指標として報告し、valid条件付きの高い占有スコアを完全再構築や全試行性能と呼ばない。元のraw/parsed/voxels・GT・入力・checkpointは不変で、全strict検算とalias感度を別欄に残した。研究API・モデル・token-count・キー照会、追加人手実験は0件。
