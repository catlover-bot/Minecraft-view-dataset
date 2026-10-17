# rest30の将来検証案：未承認・未実行

**UNAPPROVED_NOT_RUN_NO_LAUNCHER**。60建築の既存結果を見た上で作成した設計案であり、事前登録・生成承認ではない。

問いは同じO説明を使うBuilderと固定filled-boxの指標トレードオフに限定する。主estimandは建築等重みのaligned occupancy IoU差（Builder−box）。surface F1@1を必須counter-metricとし、Precision・Recall・出力失敗率・運用欠測率も併記する。

説明元は計画でOpenAIに固定済みで、従来のprovenanceと1800-capの説明生成設計を理由とする。既存の効果が有利なproviderを計算後に選ぶ規則ではない。入力が不足しても建築やproviderを黙って差し替えない。

| 候補 | 建築N | Builders | r | 将来calls | 入力予約tokens | 出力予約tokens | 予約総tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | 30 | 2 | 1 | 60 | 18000000 | 1966080 | 19966080 |
| R3 | 30 | 2 | 3 | 180 | 54000000 | 5898240 | 59898240 |

各候補の予約額は、次のBuilder別の式を足し合わせる。単価はすべて将来実行日の公式USD/1M tokensで、現在は未確認。追加課金項目や料金tierがあれば承認前に式へ反映する。

| 候補 | Builder | USD予約額の記号式 |
| --- | --- | --- |
| R1 | anthropic | `(6000000 * future_anthropic_input_USD_per_1M + 983040 * future_anthropic_output_USD_per_1M) / 1000000` |
| R1 | openai | `(12000000 * future_openai_input_USD_per_1M + 983040 * future_openai_output_USD_per_1M) / 1000000` |
| R3 | anthropic | `(18000000 * future_anthropic_input_USD_per_1M + 2949120 * future_anthropic_output_USD_per_1M) / 1000000` |
| R3 | openai | `(36000000 * future_openai_input_USD_per_1M + 2949120 * future_openai_output_USD_per_1M) / 1000000` |

予約は旧planのfull-context input＋output capに基づく保守的上限で、実入力token数・完了予測・将来のcontext適合証明ではない。料金は将来の公式単価と課金対象機能の確認が必要（REQUIRES_CURRENT_PRICE_VERIFICATION）。過去価格から新たなドル値を作らず、請求書や消費実績と区別する。

下式のfuture単価は将来実行日の公式USD/100万tokensを表す未確定変数で、値を代入していない。各候補の合計予約費用は2 providerの式の和である。別の課金対象機能があればその適用条件も別途検証する。

| 候補 | Builder | 料金予約の記号式（USD、未確定） |
| --- | --- | --- |
| R1 | anthropic | `(6000000 * future_anthropic_input_USD_per_1M + 983040 * future_anthropic_output_USD_per_1M) / 1000000` |
| R1 | openai | `(12000000 * future_openai_input_USD_per_1M + 983040 * future_openai_output_USD_per_1M) / 1000000` |
| R3 | anthropic | `(18000000 * future_anthropic_input_USD_per_1M + 2949120 * future_anthropic_output_USD_per_1M) / 1000000` |
| R3 | openai | `(36000000 * future_openai_input_USD_per_1M + 2949120 * future_openai_output_USD_per_1M) / 1000000` |

| Builder | 保存requested model ID | 入力予約/回 | 出力上限/回 | sampling |
| --- | --- | --- | --- | --- |
| anthropic | claude-haiku-4-5-20251001 | 200000 | 32768 | seed/temperature/top-p等はpayloadで省略 |
| openai | gpt-5-mini | 400000 | 32768 | seed/temperature/top-p等はpayloadで省略 |

モデルの現在の利用可能性は照会していない。省略されたsamplingパラメータの既定値や決定性は保証しない。prompt SHA：`ba7a555860628c1ad55ba5b65200ac3cf0d528be86c42f46dbb5bbdd61e6ccac`、schema SHA：`41a28cf6044541d67f794b5705b934749f94ef882732becb325d8cd1af2fdf83`。保存planから同一性を検査した。

| Builder | 指標 | eval60 valid N | SD | 将来N | r | t近似半幅の仮定範囲 |
| --- | --- | --- | --- | --- | --- | --- |
| openai | occupancy_iou | 57 | 0.048383 | 30 | 1 | [0.018066, 0.018066] |
| openai | occupancy_iou | 57 | 0.048383 | 60 | 1 | [0.012499, 0.012499] |
| openai | surface_f1_at_1 | 57 | 0.093654 | 30 | 1 | [0.034971, 0.034971] |
| openai | surface_f1_at_1 | 57 | 0.093654 | 60 | 1 | [0.024194, 0.024194] |
| anthropic | occupancy_iou | 59 | 0.041721 | 30 | 1 | [0.015579, 0.015579] |
| anthropic | occupancy_iou | 59 | 0.041721 | 60 | 1 | [0.010778, 0.010778] |
| anthropic | surface_f1_at_1 | 59 | 0.087035 | 30 | 1 | [0.032499, 0.032499] |
| anthropic | surface_f1_at_1 | 59 | 0.087035 | 60 | 1 | [0.022483, 0.022483] |
| openai | occupancy_iou | 57 | 0.048383 | 30 | 3 | [0.010431, 0.018066] |
| openai | occupancy_iou | 57 | 0.048383 | 60 | 3 | [0.007216, 0.012499] |
| openai | surface_f1_at_1 | 57 | 0.093654 | 30 | 3 | [0.020191, 0.034971] |
| openai | surface_f1_at_1 | 57 | 0.093654 | 60 | 3 | [0.013968, 0.024194] |
| anthropic | occupancy_iou | 59 | 0.041721 | 30 | 3 | [0.008995, 0.015579] |
| anthropic | occupancy_iou | 59 | 0.041721 | 60 | 3 | [0.006223, 0.010778] |
| anthropic | surface_f1_at_1 | 59 | 0.087035 | 30 | 3 | [0.018763, 0.032499] |
| anthropic | surface_f1_at_1 | 59 | 0.087035 | 60 | 3 | [0.012981, 0.022483] |

表のN60は精度を比較するための仮想参照であり、第3の実験候補ではない。元restは30建築で、N60への拡張可能性や追加コホートの利用は確認していない。

半幅はt(0.975,N−1)×eval60の建築差SD/√Nの計画近似である。0.03は許容幅として置いた値で、検証済みMCIDではない。r3の範囲は非負の建築間/反復内分散分解と条件付き独立反復を仮定し、r1半幅の1/√3〜1倍。単発データから反復内分散は識別できず、相関反復・valid選択・コホート差があれば範囲は保証されない。N30の十分性や検出力を主張しない。

将来検証を別途正当化する場合の第一候補はR1とする。R3の追加120 callsを支持する反復分散の証拠が現在はなく、建築Nも30のままである。これは実行の推奨確定・承認ではない。1 Builderへ減らせば他方の独立確認を失うため、第3候補は追加しない。

rest30の選択済み入力は30/30件がsource auditで確認済み。過去利用の証拠と未確認範囲はREST30_FEASIBILITY/SOURCESに残す。Goal005–010で未評価であっても完全未見・新分布とは呼ばない。

将来はcommon-validの同一core集合と建築内の反復平均を先に固定し、既知出力失敗のbounded failure0は別表にする。HTTP520・unknown usage・resource missingはNULLのまま維持する。従来991 callsとUSD30枠は新承認ではない。新scope・現行価格/モデル/上限の公式確認・新しい実TTY承認が別途必要で、自動retryは0。保存失敗や未知応答では停止し、過去Draftを変更しない。本Goalは台帳・authority・approval・launcherを作らず、実行コマンドも提示しない。

正常な空出力は成功分類を維持し、未定義metricを0や失敗へ置き換えない。適格反復がない建築は条件付きcommon-valid平均からのみ除外し、予定N30と実際の適格建築数・pair数をBuilder別に併記する。出力失敗率と運用欠測率は各Builderの固定30×r slotsを分母に分けて報告する。

同じO JSONから何回生成しても、image→textで失われた情報の因果効果は識別できない。入力情報への制御されたpaired介入を含む別研究が必要で、ここでは実装しない。研究API・モデル・token-count API・キー照会は0。
