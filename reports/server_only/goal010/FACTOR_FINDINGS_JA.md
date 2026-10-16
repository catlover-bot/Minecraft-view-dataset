# 説明生成元とBuilderの対応比較：全36対比

POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC。これは既知の従来結果を踏まえ、追加計算前に仕様を固定した後付け分析である。事前登録された検証、provider固有能力の因果比較、因果的な交互作用の推定ではない。

保存O説明の占有IoUでは、両BuilderともAnthropic説明−OpenAI説明が負、Gemini説明−Anthropic説明が正だった。一方、Gemini説明−OpenAI説明の95%区間は両Builderで0をまたぐ。したがってGemini対Anthropicだけを取り出して「Geminiが一貫して最良」とする解釈は支持されない。

## 完了・出典の確認

`PHASE2_RECEIPT.json` のPASS、Phase0〜2の全成果物hash、baselineのNPY参照hashを読取り検証した。元ソースの前後一致と14読取りlockは完了記録に保存されている。今回の報告作成では統計の再計算、原本変更、API送信を行っていない。

- 実行code commit: `4a3b714ad139ee0f2836d756347b6bf02dbb9a8b`。
- 仕様commit: `87f85a2b6bc06a1ae30af8e5276d3aff67c58379`、SHA256 `e52ead81c32c7b76b3f93d226af6a715d39183c14437d21c23c49b9e7f221943`。
- Phase2 receipt SHA256: `1977c24314255fc69debbe43435fff1581c19218c9f0c4f22242c1780f7cbf7d`。
- 全数JSON SHA256: `dc74a8b6ded6e9dc702960c58787a7a475551026e2c47dbae7422eab2c5244d8`。
- 公開用データ: [全36集計CSV](analysis/phase2_factors/description_builder_matched.csv)、[全設定・pairs・除外JSON](analysis/phase2_factors/description_builder_analysis.json)、[除外CSV](analysis/phase2_factors/description_builder_exclusions.csv)、[設定監査](analysis/phase2_factors/description_builder_confounds.json)。

## 対比の定義と分母

各Builderを固定し、同じ建築・同じO/J条件の説明生成元を比較した。差は表の左側providerから右側providerを引く。占有IoUと材質対応IoUは正の差が左側に有利、寸法MAEは負の差が左側に有利である。全指標は既存のaligned値を使用し、指標ごとにshiftを変えていない。

1建築1差の等重み平均で、Nは両説明からの再構築がともにvalidな建築数（57〜60）。二つのBuilderは同じ説明を繰り返し使用しており、独立Nを120や360へ増やさない。Builder別の並列表示は因果的なinteractionの識別ではない。各対比のcommon-valid建築だけをbootstrap frameとし、PCG64 seed 20261017、2000 building draws、type7 percentile 95% CIを用いた。全36対比で有限drawは2000だった。baselineのplanned-frame欠測法とは分母が異なる。新しいp値・有意性判定・Holm調整は計算していない。従来Goal007の20比較と3 familyは不変である。

## O条件の占有IoU：全6対比

| Builder | 説明生成元の差 | N | 平均差 | 95% CI |
|---|---|---:|---:|---|
| OpenAI | Anthropic − OpenAI | 57 | -0.049508 | [-0.068649, -0.030249] |
| OpenAI | Gemini − OpenAI | 57 | +0.018134 | [-0.001620, +0.038807] |
| OpenAI | Gemini − Anthropic | 60 | +0.069926 | [+0.048026, +0.092757] |
| Anthropic | Anthropic − OpenAI | 59 | -0.059459 | [-0.077176, -0.040266] |
| Anthropic | Gemini − OpenAI | 59 | +0.003861 | [-0.013684, +0.023089] |
| Anthropic | Gemini − Anthropic | 60 | +0.061978 | [+0.040345, +0.083000] |

OpenAI BuilderのGemini−OpenAIは+0.018134、Anthropic Builderでは+0.003861で、両区間は0を含む。Anthropic−OpenAIはそれぞれ−0.049508、−0.059459、Gemini−Anthropicは+0.069926、+0.061978である。これは固定した成功ペアにおける設定込みの観測差であり、除外された失敗を含めた全入力の平均差ではない。

J条件でもAnthropic−OpenAIは両Builderで負（−0.055922、−0.057014）、Gemini−Anthropicは正（+0.070119、+0.053275）だった。Gemini−OpenAIはOpenAI Builderで+0.015936、Anthropic Builderで−0.003739で、やはり両区間が0を含む。大まかな方向の対応は確認できるが、Jは同じtyped JSON情報の別API生成で、独立した説明コホートによる再現実験ではない。

寸法MAEでは、全O/J・両BuilderでAnthropic−OpenAIが正（約+5.02〜+5.55）、Gemini−OpenAIが負（約−3.79〜−4.58）、Gemini−Anthropicが負（約−9.04〜−9.60）である。占有IoUの差と寸法誤差の差は同一の問いではない。材質対応IoUではGemini−Anthropicは全4条件で正、Gemini−OpenAIは全4条件で区間が0を含む。Anthropic−OpenAIの材質差は全4条件で負の点推定だが、Anthropic BuilderのO条件は区間が0を含む。材質集合F1を座標上の材質一致の代用にしていない。

## 全36対比（省略なし）

CSV/JSONは丸め前の値を保持する。以下は小数6桁。区間が0を含むかどうかの記述は、新しい検定や多重性を調整した判定ではない。

| Builder | 条件 | 説明生成元の差 | aligned指標 | N | 平均差 | 95% CI |
|---|---|---|---|---:|---:|---|
| OpenAI | O | Anthropic − OpenAI | 占有IoU ↑ | 57 | -0.049508 | [-0.068649, -0.030249] |
| OpenAI | O | Anthropic − OpenAI | 寸法MAE ↓ | 57 | +5.356725 | [+4.233772, +6.462281] |
| OpenAI | O | Anthropic − OpenAI | 材質対応IoU ↑ | 57 | -0.010686 | [-0.019237, -0.002450] |
| OpenAI | O | Gemini − OpenAI | 占有IoU ↑ | 57 | +0.018134 | [-0.001620, +0.038807] |
| OpenAI | O | Gemini − OpenAI | 寸法MAE ↓ | 57 | -3.888889 | [-5.082164, -2.719298] |
| OpenAI | O | Gemini − OpenAI | 材質対応IoU ↑ | 57 | +0.004042 | [-0.003566, +0.011639] |
| OpenAI | O | Gemini − Anthropic | 占有IoU ↑ | 60 | +0.069926 | [+0.048026, +0.092757] |
| OpenAI | O | Gemini − Anthropic | 寸法MAE ↓ | 60 | -9.338889 | [-10.516806, -8.116667] |
| OpenAI | O | Gemini − Anthropic | 材質対応IoU ↑ | 60 | +0.014262 | [+0.006245, +0.023104] |
| OpenAI | J | Anthropic − OpenAI | 占有IoU ↑ | 57 | -0.055922 | [-0.074586, -0.037114] |
| OpenAI | J | Anthropic − OpenAI | 寸法MAE ↓ | 57 | +5.362573 | [+4.292398, +6.438743] |
| OpenAI | J | Anthropic − OpenAI | 材質対応IoU ↑ | 57 | -0.010074 | [-0.018281, -0.002671] |
| OpenAI | J | Gemini − OpenAI | 占有IoU ↑ | 57 | +0.015936 | [-0.000350, +0.032659] |
| OpenAI | J | Gemini − OpenAI | 寸法MAE ↓ | 57 | -3.801170 | [-5.041082, -2.584649] |
| OpenAI | J | Gemini − OpenAI | 材質対応IoU ↑ | 57 | +0.004654 | [-0.005397, +0.014494] |
| OpenAI | J | Gemini − Anthropic | 占有IoU ↑ | 60 | +0.070119 | [+0.052434, +0.088347] |
| OpenAI | J | Gemini − Anthropic | 寸法MAE ↓ | 60 | -9.044444 | [-10.222500, -7.844444] |
| OpenAI | J | Gemini − Anthropic | 材質対応IoU ↑ | 60 | +0.015196 | [+0.006728, +0.024394] |
| Anthropic | O | Anthropic − OpenAI | 占有IoU ↑ | 59 | -0.059459 | [-0.077176, -0.040266] |
| Anthropic | O | Anthropic − OpenAI | 寸法MAE ↓ | 59 | +5.548023 | [+4.491384, +6.559463] |
| Anthropic | O | Anthropic − OpenAI | 材質対応IoU ↑ | 59 | -0.004878 | [-0.010428, +0.000322] |
| Anthropic | O | Gemini − OpenAI | 占有IoU ↑ | 59 | +0.003861 | [-0.013684, +0.023089] |
| Anthropic | O | Gemini − OpenAI | 寸法MAE ↓ | 59 | -3.790960 | [-4.988701, -2.610028] |
| Anthropic | O | Gemini − OpenAI | 材質対応IoU ↑ | 59 | +0.002597 | [-0.005642, +0.010828] |
| Anthropic | O | Gemini − Anthropic | 占有IoU ↑ | 60 | +0.061978 | [+0.040345, +0.083000] |
| Anthropic | O | Gemini − Anthropic | 寸法MAE ↓ | 60 | -9.461111 | [-10.650139, -8.183056] |
| Anthropic | O | Gemini − Anthropic | 材質対応IoU ↑ | 60 | +0.007292 | [+0.000993, +0.014308] |
| Anthropic | J | Anthropic − OpenAI | 占有IoU ↑ | 60 | -0.057014 | [-0.076485, -0.037523] |
| Anthropic | J | Anthropic − OpenAI | 寸法MAE ↓ | 60 | +5.022222 | [+3.694306, +6.261111] |
| Anthropic | J | Anthropic − OpenAI | 材質対応IoU ↑ | 60 | -0.007069 | [-0.013963, -0.000417] |
| Anthropic | J | Gemini − OpenAI | 占有IoU ↑ | 60 | -0.003739 | [-0.020882, +0.014086] |
| Anthropic | J | Gemini − OpenAI | 寸法MAE ↓ | 60 | -4.577778 | [-6.200000, -3.127500] |
| Anthropic | J | Gemini − OpenAI | 材質対応IoU ↑ | 60 | +0.002053 | [-0.004794, +0.008705] |
| Anthropic | J | Gemini − Anthropic | 占有IoU ↑ | 60 | +0.053275 | [+0.032933, +0.073885] |
| Anthropic | J | Gemini − Anthropic | 寸法MAE ↓ | 60 | -9.600000 | [-10.988889, -8.160972] |
| Anthropic | J | Gemini − Anthropic | 材質対応IoU ↑ | 60 | +0.009123 | [+0.003093, +0.015144] |

## 既知失敗とglobal942の欠測

予定2160対比×建築×指標行のうち、保存された有効paired行は2118、除外記録は42。42は重複する対比・指標ごとの除外数であり、42件の独立失敗ではない。元の除外対象は次の7 selected slots（validation failure 6件、unknown-usage HTTP520 1件）である。いずれもOpenAI説明を入力とする観測だった。

| global index | 建築 | Builder | 条件 | 説明生成元 | 状態 |
|---:|---|---|---|---|---|
| 88 | 078 | OpenAI | J | OpenAI | validation failure / null |
| 144 | 062 | OpenAI | J | OpenAI | validation failure / null |
| 158 | 062 | OpenAI | O | OpenAI | validation failure / null |
| 598 | 054 | OpenAI | O | OpenAI | validation failure / null |
| 670 | 053 | OpenAI | O | OpenAI | validation failure / null |
| 841 | 011 | Anthropic | O | OpenAI | validation failure / null |
| 942 | 085 | OpenAI | J | OpenAI | unknown usage / HTTP520 / null |

OpenAI BuilderのOでは053/054/062の3建築、Jでは062/078の既知失敗と085（global942）の未知欠測が、OpenAI説明を含む2対比から除外される。Anthropic BuilderのOでは011の1建築を同様に除外し、同BuilderのJでは除外なし。Gemini−Anthropicにはこれらの欠測がなく全条件N=60である。common-valid集合が対比ごとに異なるため、三つの平均差の加法性を仮定しない。

global942はJ/OpenAI Builder/OpenAI説明のoperational_unmeasurableとして残り、metrics・usage・responseはnullのままである。0への補完、認証retryとの混同、欠測解消はしていない。global940の承認retryは別のO slotに関する履歴で、成功を一度だけ選択する従来規則を変更していない。

## 交絡と研究上の限定

Phase0では全180 requestが同じ建築あたり同一bytes・順序の6枚の画像と同一prompt hashesを3 description providers間で共有することを確認した。12は保存候補view数であり、送信画像数ではない。画像・promptの一致は以下の設定差を消さない。

| 説明生成元 | 保存model ID | max_tokens | 保存temperature | requestレコード作成UTC範囲 |
|---|---|---:|---:|---|
| OpenAI | gpt-5-mini | 1800 | 0.2 | 2026-02-28T15:03:32.530792+00:00 — 2026-02-28T15:29:34.888485+00:00 |
| Anthropic | claude-haiku-4-5-20251001 | 1800 | 0.2 | 2026-03-01T04:49:59.031522+00:00 — 2026-03-01T05:04:35.457764+00:00 |
| Gemini | gemini-3.1-pro-preview | 4096 | 0.2 | 2026-04-18T15:35:51.120649+00:00 — 2026-04-18T19:44:27.411572+00:00 |

OpenAI/Anthropic説明の上限1800に対しGeminiは4096であり、model/version、出力予算、実施時期、providerごとのsampling実装が同時に異なる。created_atは応答後に作成された保存requestレコードの時刻で、送信開始時刻ではない。top_pとseedは記録されず、Geminiのllm_seed=-1も実際の固定sampling seedの証明ではない。description requestのtemperature記録0.2を、別段階のBuilder生成のtemperatureが固定された証拠に転用しない。

この結果は「特定の保存説明と生成設定が、同じ建築の再構築結果にどう対応したか」という観測比較を支える。説明生成元の純粋な優劣、Builderとの因果的interaction、自然言語表記のみの因果効果は識別しない。既知失敗を除いた選択と単発生成、同じ60建築の反復、後付け探索の多重性を明示して解釈する。
