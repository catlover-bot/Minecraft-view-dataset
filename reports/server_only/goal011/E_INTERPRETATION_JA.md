# Eの置換パターン：全件結果の読解

POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC。出典は `E_CASES_101.csv`、`E_MAPPING_UPPER_BOUNDS.csv`、`E_FAILURE_MECHANISM_AUDIT.json`。元の101 valid出力（OpenAI55、Anthropic46）を全件検算し、19 truncation（5／14）を別に残した。未計算0。各Builder内の成功建築Nは55／46であり、101を独立建築Nとして扱わない。全101件の保存shiftは0で、absoluteとalignedは一致した。

| 元出力の性質 | OpenAI | Anthropic |
| --- | --- | --- |
| 占有geometryが完全一致 | 36/55 | 14/46 |
| その完全geometry例のうち材質不一致あり | 22/36 | 4/14 |
| geometryと厳密材質がともに完全一致 | 14/55 | 10/46 |
| 共通占有内に材質不一致がある | 36/55 | 27/46 |

共通占有内の材質不一致がある建築だけで、GT-informed一対一／many-to-one上界が説明し得る不一致割合を建築等重みに平均すると、OpenAI 24.5833%／42.5653%（N36）、Anthropic 47.9229%／50.6037%（N27）だった。不一致0の各19件ではこの割合はNULLである。

voxelを合算すると、OpenAIの元不一致15,997のうち9,720／11,899、Anthropicの26,894のうち20,871／21,494が対応上界で説明可能になる。これは大きな不一致を持つケースに強く重みが掛かる集計で、建築等重み平均とは違う。全体を「主に単純なラベル交換」と一括分類する根拠にはしない。

事前固定した `occupancy IoU≥0.99` かつ `material IoU<0.5` の全8件を示す。表の上界はGTを見て各予測別に最適化した値で、モデル予測・補正後の性能ではない。

| building / Builder | occupancy IoU | 元material IoU | C | 元M | 一対一M上界 | many-to-one M上界 | 共通占有の残差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 016 / Anthropic | 0.995499 | 0.280488 | 3981 | 1748 | 3776 | 3776 | 205 |
| 026 / OpenAI | 1 | 0.264392 | 4206 | 1759 | 4098 | 4098 | 108 |
| 034 / OpenAI | 1 | 0.223412 | 4912 | 1794 | 4752 | 4752 | 160 |
| 047 / Anthropic | 0.991091 | 0.203410 | 4005 | 1360 | 4002 | 4002 | 3 |
| 059 / Anthropic | 0.999816 | 0.427388 | 5425 | 3249 | 5101 | 5101 | 324 |
| 064 / Anthropic | 0.999651 | 0.471397 | 5722 | 3667 | 5698 | 5698 | 24 |
| 065 / Anthropic | 1 | 0.121442 | 4511 | 977 | 4463 | 4463 | 48 |
| 095 / Anthropic | 0.990046 | 0.410347 | 2387 | 1396 | 1655 | 1655 | 732 |

034/OpenAIは g=p=C=4912、M=1794、不一致3118で、元material-aware IoUは `1794/(4912+4912−1794)=0.2234122042`。GT `sandstone` の2958セルは予測 `glass`、GT `glass` の160セルは予測 `brick_block` だった。後者は正しい `brick_block` 568セルと同じ出力IDに合流している。

予測 `glass`→GT `sandstone`、予測 `brick_block`→GT `brick_block`、残りの `quartz_block`・`sea_lantern`・`stone_slab` は同名という最適対応で、両上界Mは4752になる。この対応は同点なしで、元不一致の2958/3118（94.8685%）を説明するが160セルを分離できない。上界material IoUは0.9369085174で、元の0.2234122042を置換しない。したがって、可逆的なpure label swapではない。

034では428/428のinclusive action境界が一意かつ非重複で一致し、202対の材質一致・226対の不一致を確認した。GT `sandstone`→出力 `glass` は140操作、GT `glass`→出力 `brick_block` は86操作である。voxel数と操作数を混同しない。出力上の系統的置換・合流の証拠は強いが、モデル内部でどの認知的誤りが起きたかは識別できない。

一対一mappingの同点はOpenAI4件（016/027/042/094）、Anthropic2件（026/053）。many-to-oneの同点は0件。全最適解の数そのものは列挙せず `optimal_mapping_count=null` とし、各予測ラベルの全最適解における候補targetを保存した。C=0、未計算mapping、未観測予測labelは今回0件だが、fixtureではNULL・欠落・rare classも検証している。

曖昧なreference actionを含むケースは22/55・33/46、未分類predicted actionを含むケースは17/55・31/46だった。対応不能な境界を、推測によって「shift」「merge」「omission」という意図に分類しない。元のgeometry差、ラベル合流、局所的な残余不一致を残し、GTを使う上界を新しい成功件数へ加算しない。
