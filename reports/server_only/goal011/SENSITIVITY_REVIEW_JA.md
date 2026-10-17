# 固定感度表の独立読解

POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC。出典は `global_saved_effects.csv` と `quintile_effects.csv`。既存行を読むための補足で、新指標・CI・p値を追加しない。差はすべて Builder−対照である。

O／filled-box／aligned／全説明平均では、両Builderとも5サイズ層すべてで IoU・Recall 差が負、Precision・surface F1@1 差が正だった。各層の独立Nは12建築である。平均の符号が全建築の符号を意味するわけではない。例えば OpenAI の層2は IoU差平均 −0.000537 だが、建築ごとの符号は正7・負5だった。

説明provider別のIoUでは順序が変わる。

| 説明provider | OpenAI Builder：差／建築N | Anthropic Builder：差／建築N |
| --- | --- | --- |
| openai | −0.026649／57 | −0.018488／59 |
| anthropic | −0.038612／60 | −0.037646／60 |
| gemini | ＋0.027921／60 | ＋0.020939／60 |

このprovider別の符号はそれぞれ全5サイズ層でも維持された。「どの説明でも箱が上回る」という主張は支持しない。説明には寸法値そのものも含まれるため、Builderだけでなく同じ説明を使うbaselineも変化する。providerのモデルversion、出力予算、生成時期等も異なり、provider固有能力の因果差とは解釈しない。

surface F1@1も一様ではない。Anthropic説明での差は ＋0.014792／＋0.001249 で、保存済みCIは双方0を含む。Gemini説明では ＋0.209307／＋0.156243 だった。CIはGoal010v2の探索的な保存値で、独立した有意な発見の数として扱わない。

shell対照にすると、alignedのIoU・Recall差は両Builderで全5層正になる。一方、Anthropic Builderのsurface F1@1差は層3で −0.001973、層4で −0.015039。全体平均 ＋0.022389 を全サイズ層へ一般化できない。

Precisionの正の差は評価frameにも依存する。filled-box／全説明平均のabsolute差は OpenAI −0.015776、Anthropic −0.017812 で、alignedでは ＋0.092574／＋0.065285 となる。absoluteのsurface差は ＋0.022371／＋0.014630 だが、Anthropicの保存済みCIは [−0.000746, ＋0.029833]、層0のabsolute surface差は両Builderで負だった。GT alignmentは実配置能力の検証ではない。

core4指標でも感度を含む6指標でも、有効mask差異は160群中0だった。footprintは層・providerで符号が混在し、全体の保存済みCIも0を含む。細分した表は小標本の記述に限定し、最も有利なprovider・対照・frameを選び直さない。
