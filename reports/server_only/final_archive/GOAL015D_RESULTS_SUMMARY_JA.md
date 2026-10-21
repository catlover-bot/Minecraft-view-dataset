# Goal015D実測16件：保存値の転記

Goal015Dは準備READMEの後に実行され、保存済み16行では **Aの非空0/8、Bの非空8/8** だった。ここでは利用者指定のsummaryと保存証明のSHA256、およびsummaryが束縛するevaluation・plan・ledgerのbytes/hashを確認し、既存の判定と数値を転記した。再生成・再評価・新しい統計量やCIの計算は行っていない。

| 条件 | 予定／SETTLED | protocol-valid | 正常empty | 非empty | 既知出力失敗 |
|---|---:|---:|---:|---:|---:|
| A：現在のproperties順 | 8/8 | 5 | 5 | 0 | 3 |
| B：元のproperties順 | 8/8 | 8 | 0 | 8 | 0 |
| 合計 | 16/16 | 13 | 5 | 8 | 3 |

Aの正常empty5件はcarve-onlyだった。失敗3件はAnthropic Aの`goal015d_call_005`、`009`、`016`で、保存判定はすべてtruncation。出力失敗のgridと指標はNULLのままにした。正常emptyのoccupancy IoU等の0は元評価器の定義による保存値であり、失敗へ0を代入したものではない。正常emptyのprecisionは定義不能のNULLを維持した。

| 建築 | Builder | A：非empty/予定 | B：非empty/予定 | Aの内訳 |
|---|---|---:|---:|---|
| building_082 | OpenAI | 0/2 | 2/2 | 正常empty2 |
| building_082 | Anthropic | 0/2 | 2/2 | 正常empty1、truncation1 |
| building_003 | OpenAI | 0/2 | 2/2 | 正常empty2 |
| building_003 | Anthropic | 0/2 | 2/2 | truncation2 |

既知usageは全16件で、保存済み実費合計は **USD0.67051700**。提案上限USD5.00や保守予約USD4.235008を実費とは扱わない。親S/FのUSD4.43765700、Goal015CのUSD0.01865750とも別scopeである。

各16行の状態・操作数・voxel数・usage・費用と、既定6指標の保存値は [GOAL015D_RESULTS_16.csv](GOAL015D_RESULTS_16.csv) にある。指標はaligned occupancy IoU、surface F1@1、occupancy precision/recall、dimension MAE、material-aware IoUで、元occupancy最適shiftを共有する。NULLは文字列`NULL`で示す。元の8セル表と8組のB−Aはcanonical evaluation directoryの`cells_8.csv`と`pairs_8.json`に残る。

この実建築2件・各2反復では、properties順の介入に伴う非空生成の切替を観測した。非空は再建築の正確さや外観・機能の妥当性を意味しない。対象は利用済みbuilding_082/003のdevelopment canaryで、未見holdoutでも、本体60建築の追加標本でもない。Goal015Cの合成課題1種類・12呼出しとも別の観測である。

Bはop位置だけでなく座標keys等の並びも戻すため、その個別効果は未分離。provider内部のbug、一般的成功率、モデル間優劣、旧72件の原因や反実仮想は確定していない。旧S/Fは72件送信・8件未送信の未完状態を保持し、Dの結果で補完・訂正しない。

出典は `/root/minecraft-runs/goal015d-20261020T114450Z-ebe767ac/evaluations/goal015d-20261020T125031610548Z-d17a0619/`。summary SHA256は `e81c776ac9ec66df9846b7baaa25d766e4d1525c8498ff3898ea127e7b15aaed`、evaluation SHA256は `90cab42c5523fa253f1c2275002b10e26ad091ce7763e719ab0d85fb641a3960`。

既存未公開Draft **418394657**／asset **659324320**、archive SHA256 `c9862ff6c6b8dc899d9ec69278426d7e93472f1ea2452e508fb161c7655c9227`。保存証明SHA256は `6eff57fb1180221fa63a1450915cd732330218740aa3a89674e13a50d1bd07e9`。これは既存証明の照合であり、本転記作業でのGitHub再取得を意味しない。今回の最終退避の取得・復元結果は別の最終保存receiptに従う。[転記元と照合記録](GOAL015D_RESULTS_SOURCE.json) を参照。
