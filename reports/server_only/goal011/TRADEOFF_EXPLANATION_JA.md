# O条件：coverage と selectivity のトレードオフ監査

POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC

既存の6指標だけを使用し、全60の95% CIはGoal010v2から変更せず再利用した。五分位の新集計は記述のみで、新CI・p値・bootstrapを計算しない。

差はBuilder−対照。building内の適格な説明生成元ペアを平均した後、建築を等重みで平均する。2 Buildersと3説明を独立Nとして足さない。alignedはGTに対するoccupancy最適シフト後の評価であり、世界原点の推定精度ではない。

| Builder | 対照 | frame | 指標 | ペアN / 建築N | 差 [保存済み95% CI] |
| --- | --- | --- | --- | --- | --- |
| openai | filled_box | absolute | occupancy_iou | 177 / 60 | -0.047421 [-0.054037, -0.040267] |
| openai | filled_box | absolute | occupancy_precision | 177 / 60 | -0.015776 [-0.024309, -0.006718] |
| openai | filled_box | absolute | occupancy_recall | 177 / 60 | -0.238957 [-0.266245, -0.211470] |
| openai | filled_box | absolute | surface_f1_at_1 | 177 / 60 | +0.022371 [+0.011046, +0.034417] |
| openai | filled_box | absolute | surface_f1_at_2 | 177 / 60 | +0.024363 [+0.009190, +0.040261] |
| openai | filled_box | absolute | footprint_iou | 177 / 60 | -0.003034 [-0.014740, +0.010079] |
| openai | filled_box | aligned | occupancy_iou | 177 / 60 | -0.011960 [-0.020352, -0.003229] |
| openai | filled_box | aligned | occupancy_precision | 177 / 60 | +0.092574 [+0.069295, +0.111379] |
| openai | filled_box | aligned | occupancy_recall | 177 / 60 | -0.211233 [-0.234255, -0.185203] |
| openai | filled_box | aligned | surface_f1_at_1 | 177 / 60 | +0.099288 [+0.083128, +0.116816] |
| openai | filled_box | aligned | surface_f1_at_2 | 177 / 60 | +0.091993 [+0.074742, +0.110518] |
| openai | filled_box | aligned | footprint_iou | 177 / 60 | +0.005806 [-0.004320, +0.017842] |
| openai | six_face_shell | absolute | occupancy_iou | 177 / 60 | +0.026325 [+0.020923, +0.031746] |
| openai | six_face_shell | absolute | occupancy_precision | 177 / 60 | +0.007532 [-0.004013, +0.018511] |
| openai | six_face_shell | absolute | occupancy_recall | 177 / 60 | +0.064527 [+0.050160, +0.079393] |
| openai | six_face_shell | absolute | surface_f1_at_1 | 177 / 60 | +0.022371 [+0.011046, +0.034417] |
| openai | six_face_shell | absolute | surface_f1_at_2 | 177 / 60 | +0.024363 [+0.009190, +0.040261] |
| openai | six_face_shell | absolute | footprint_iou | 177 / 60 | -0.003034 [-0.014740, +0.010079] |
| openai | six_face_shell | aligned | occupancy_iou | 177 / 60 | +0.056642 [+0.047668, +0.066382] |
| openai | six_face_shell | aligned | occupancy_precision | 177 / 60 | +0.009490 [-0.016272, +0.032339] |
| openai | six_face_shell | aligned | occupancy_recall | 177 / 60 | +0.109668 [+0.092822, +0.126782] |
| openai | six_face_shell | aligned | surface_f1_at_1 | 177 / 60 | +0.057884 [+0.044175, +0.071634] |
| openai | six_face_shell | aligned | surface_f1_at_2 | 177 / 60 | +0.063459 [+0.047976, +0.079435] |
| openai | six_face_shell | aligned | footprint_iou | 177 / 60 | +0.032533 [+0.021837, +0.044071] |
| anthropic | filled_box | absolute | occupancy_iou | 179 / 60 | -0.036624 [-0.043680, -0.029706] |
| anthropic | filled_box | absolute | occupancy_precision | 179 / 60 | -0.017812 [-0.027318, -0.008484] |
| anthropic | filled_box | absolute | occupancy_recall | 179 / 60 | -0.179492 [-0.207438, -0.152618] |
| anthropic | filled_box | absolute | surface_f1_at_1 | 179 / 60 | +0.014630 [-0.000746, +0.029833] |
| anthropic | filled_box | absolute | surface_f1_at_2 | 179 / 60 | +0.014670 [-0.005343, +0.033137] |
| anthropic | filled_box | absolute | footprint_iou | 179 / 60 | -0.004689 [-0.015749, +0.005193] |
| anthropic | filled_box | aligned | occupancy_iou | 179 / 60 | -0.011615 [-0.019086, -0.004501] |
| anthropic | filled_box | aligned | occupancy_precision | 179 / 60 | +0.065285 [+0.049145, +0.080604] |
| anthropic | filled_box | aligned | occupancy_recall | 179 / 60 | -0.153559 [-0.176890, -0.129602] |
| anthropic | filled_box | aligned | surface_f1_at_1 | 179 / 60 | +0.064520 [+0.048611, +0.081275] |
| anthropic | filled_box | aligned | surface_f1_at_2 | 179 / 60 | +0.055132 [+0.037197, +0.073300] |
| anthropic | filled_box | aligned | footprint_iou | 179 / 60 | +0.000194 [-0.006538, +0.008030] |
| anthropic | six_face_shell | absolute | occupancy_iou | 179 / 60 | +0.037482 [+0.031008, +0.043908] |
| anthropic | six_face_shell | absolute | occupancy_precision | 179 / 60 | +0.006480 [-0.005626, +0.018276] |
| anthropic | six_face_shell | absolute | occupancy_recall | 179 / 60 | +0.125632 [+0.105078, +0.146541] |
| anthropic | six_face_shell | absolute | surface_f1_at_1 | 179 / 60 | +0.014630 [-0.000746, +0.029833] |
| anthropic | six_face_shell | absolute | surface_f1_at_2 | 179 / 60 | +0.014670 [-0.005343, +0.033137] |
| anthropic | six_face_shell | absolute | footprint_iou | 179 / 60 | -0.004689 [-0.015749, +0.005193] |
| anthropic | six_face_shell | aligned | occupancy_iou | 179 / 60 | +0.057066 [+0.048953, +0.065622] |
| anthropic | six_face_shell | aligned | occupancy_precision | 179 / 60 | -0.017976 [-0.040095, +0.003110] |
| anthropic | six_face_shell | aligned | occupancy_recall | 179 / 60 | +0.168885 [+0.149146, +0.189727] |
| anthropic | six_face_shell | aligned | surface_f1_at_1 | 179 / 60 | +0.022389 [+0.008729, +0.035419] |
| anthropic | six_face_shell | aligned | surface_f1_at_2 | 179 / 60 | +0.026303 [+0.010538, +0.042302] |
| anthropic | six_face_shell | aligned | footprint_iou | 179 / 60 | +0.027073 [+0.019425, +0.035461] |

五分位×Builder×生成元scope×shape×frameのvalid-mask監査は160群。4 core指標間で異なる群は0。6指標間で異なる群は0。新しい五分位表では常に6指標の共通valid集合を使い、追加除外を記録した。

全体表でも4 core指標の適格pair ID集合は一致する。Builder別・説明scope別の適格Nは保持し、Builder間で同じNとは仮定しない。

filled boxはGT占有を広く覆う一方で余分な占有を作る。Builderは占有を選別し、その分作り漏らしも残り得る。これらはPrecision・Recall・IoUを併記して検討する。boundary-neighbor F1は6近傍境界セル中心の距離で、空洞内壁を含み、撮影画像の可視面・見た目・建築機能の忠実度ではない。

五分位は最大12建築で、細分した生成元比較の不確実性をこの記述表だけで解消できない。正負の件数を独立の有意な発見数と数えず、全生成元・両Builder・両shape・両frameを保持する。正常emptyや未定義scalarを既知失敗0に混ぜず、HTTP520は未計測のまま残す。

将来設計のSDとt近似半幅はeval60のvalid-only建築差から得た設計入力である。N30/N60とr1/r3の範囲は反復内・建築間分散の非負分解と独立反復を仮定する。反復分散は未同定で、十分な標本数・検出力・将来CIを保証しない。目標半幅0.03は計画上の許容幅でMCIDではない。実行承認はなく、研究API送信0。

| 条件 | Builder | 固定slot N | 成功 | 既知出力失敗 | 運用欠測 | 正常empty | 既知失敗率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| O | openai | 180 | 177 | 3 | 0 | 0 | 0.016667 |
| O | anthropic | 180 | 179 | 1 | 0 | 0 | 0.005556 |
| J | openai | 180 | 177 | 2 | 1 | 0 | 0.011111 |
| J | anthropic | 180 | 180 | 0 | 0 | 1 | 0.000000 |
| P | openai | 60 | 60 | 0 | 0 | 0 | 0.000000 |
| P | anthropic | 60 | 60 | 0 | 0 | 0 | 0.000000 |
| E | openai | 60 | 55 | 5 | 0 | 0 | 0.083333 |
| E | anthropic | 60 | 46 | 14 | 0 | 0 | 0.233333 |

失敗率の分母は固定slotであり、failed attemptを独立建築として追加しない。global940は承認済みretryの成功を1slotとして選択し、元401とglobal942のHTTP520に対するunknown-usage holdは維持する。費用やusageの推定をしない。

O baselineの構築入力は保存説明のdimensions_estimateと固定原点(0,0,0)、Pは開示world_bboxだけである。いずれも人工材質stoneを材質比較に用いない。構築時の入力制約と、Builder同様にGTを参照するaligned評価を区別する。

旧Goal010とv2のIoU差に許容内の浮動小数点丸め差がある場合はclaim auditに保存する。数値の近似一致をbit完全一致と呼ばず、過去表を書き換えない。相関する指標と探索的CIを独立した有意な発見として数えない。
