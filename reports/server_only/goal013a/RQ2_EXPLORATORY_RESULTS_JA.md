Goal013Aは、保存summaryの画像整合度を実採点し、計算前に固定した関連分析まで完了した。以下は `POST_HOC_EXPLORATORY` であり、通常のcaption品質RQ2や因果効果の検証完了を意味しない。

全体と各description providerの代表性gateがPASSした後、score freeze SHA `b014460fc3119c2ba6066f7e87d4bf5ba09cef3390f4be57c47729cb205b3ae0` を検証して初めて固定O結果を結合した。主結果はvalid-only、既知モデル出力失敗だけIoU=0にする既定感度分析を別行で示す。正常empty success・未定義値・運用欠測・score欠測を失敗へ再分類していない。

| policy | Builder target | paired記述行 | 寄与建築 | Spearman rho | 探索的95% CI | failure0行 |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| valid_only | openai | 175 | 60 | -0.07692 | [-0.22561, 0.06731] | 0 |
| valid_only | anthropic | 177 | 60 | -0.12389 | [-0.26488, 0.01379] | 0 |
| valid_only | both_builders_mean | 174 | 60 | -0.10831 | [-0.25426, 0.02977] | 0 |
| known_failure0 | openai | 178 | 60 | -0.10526 | [-0.25548, 0.04060] | 3 |
| known_failure0 | anthropic | 178 | 60 | -0.11724 | [-0.26047, 0.01976] | 1 |
| known_failure0 | both_builders_mean | 178 | 60 | -0.11721 | [-0.26260, 0.02565] | 4 |

valid-onlyの実共通集合はOpenAI 175、Anthropic 177、両方valid平均174行。すべて60建築が寄与する。180予定行や178採点成功をそのまま関連分析Nへ流用していない。2 Buildersを独立N360とは扱わない。pooledの3主推定は弱い負の値で区間が0を含む。この結果は、このcohortでsummary画像整合度がaligned occupancy IoUの明瞭な順位指標になっているという根拠を与えない。関連が存在しないことの証明でもない。

| policy | Builder | 記述行 | 寄与建築 | 建築内一致率の建築等重み平均 | 探索的95% CI | comparable pairs | 一致 / 不一致 | missing pairs |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- | ---: |
| valid_only | openai | 175 | 59 | 0.44633 | [0.36207, 0.52874] | 171 | 79 / 92 | 9 |
| valid_only | anthropic | 177 | 60 | 0.46111 | [0.38333, 0.53889] | 174 | 81 / 93 | 6 |
| known_failure0 | openai | 178 | 60 | 0.45556 | [0.37778, 0.53889] | 176 | 80 / 96 | 4 |
| known_failure0 | anthropic | 178 | 60 | 0.45000 | [0.37778, 0.52778] | 176 | 81 / 95 | 4 |

各行のplanned candidate pairsは180（60建築×3 description-provider対）。score-only/outcome-only/both tieはすべて0。点推定は建築ごとの一致率を等重みで平均した値なので、全一致pair数÷全comparable pair数とは一般に異なる。valid-only OpenAIは1建築が比較可能pairを持たず寄与59建築。感度表で正常emptyや運用欠測を0にして比較対を増やす処理はない。

| token長補助診断 | paired記述行 | 寄与建築 | score対非切捨てsummary token長のrho | 探索的95% CI |
| --- | ---: | ---: | ---: | --- |
| all | 178 | 60 | 0.31124 | [0.16949, 0.44814] |
| openai | 60 | 60 | 0.16998 | [-0.10676, 0.43358] |
| anthropic | 60 | 60 | 0.26987 | [0.01471, 0.50838] |
| gemini | 58 | 58 | 0.18285 | [-0.09411, 0.45187] |

pooled token長診断は0.31124で、scoreと記述長の関連が見える。provider構成や内容との交絡を除去した推定ではなく、長く書けば品質が改善することや、scoreを事後補正すべきことを示さない。これ以外の下位群、p値、Holm family、GT-proxy探索は追加していない。

14 estimandsはすべて同じplanned 60建築frameからPCG64 seed20261017の2000 cluster drawsを使用し、各drawの建築多重度を保持した。pooledでは毎draw順位を再計算した。全14行でfinite2000/undefined0、寄与建築数2以上なので既定CI gateを満たす。CIはtype7 percentile 95%で、探索的・finite draw条件付き。全drawはcanonical `associate/ASSOCIATIONS_FULL.json` に保存し、公開summaryはそのSHAを参照する。

今回のscorer入力は原summaryと保存requestの6画像のみ。Builderはsummary以外を含むO JSON全文を読んでおり、入力情報範囲が異なる。低相関から全文descriptionが役立たないとはいえない。GT非参照scoreでも建築・provider等を共有するので統計的独立は保証されない。高相関が得られていたとしても因果効果、独立外部予測性能、人間品質妥当性の証明にはならない。

CLIPScore単画像式に本研究のmean6集約を適用したもので、検証済みMinecraft multi-view品質尺度とは呼ばない。画像は960×540から398×224へresizeし、224×224へcenter cropするため面積56.28%が残る。建築全体がcrop内に残るか、遮蔽や可視面の正しさは未検証。59/178記述にtoken切捨てがあり、今回gate PASSは全文をモデルが読めたという意味ではない。

旧RQ2は未検証のまま保持する。今回完了したのは、保存summaryの画像整合度の計算可能性検証・実採点と、この60建築に限る固定探索関連分析である。人手妥当性、外部cohort、全文O品質、言語化の情報損失の因果主張、美観・機能・建築としての忠実度は確立していない。追加のpaid生成やpost-hoc指標追加は行わず、今回の範囲で終了する。
