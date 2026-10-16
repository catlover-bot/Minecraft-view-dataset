# Goal007：最終統合解析の要点

分析ラベルは **fixed60 analysis with one operational missing observation**。固定60建築・960 slots（O/J各360、P/E各120）を統合した。selected完了応答は959、そのうちモデル出力失敗25。aligned IoUのvalid観測は934、bounded IoUのfailure0適格は959、運用上の未測定slotは1。固定対象の試行履歴961件と歴史的送信991件を区別し、本解析の研究API送信は0回。

| Builder | Condition | IoU (valid slots/buildings) | MAE (valid slots/buildings) | F0 IoU (eligible slots) |
|---|---|---:|---:|---:|
| OpenAI | O | 0.1760 (177/60) | 12.8259 (177/60) | 0.1735 (180) |
| OpenAI | J | 0.1712 (177/60) | 13.1639 (177/60) | 0.1696 (179) |
| OpenAI | P | 0.1674 (60/60) | 4.4000 (60/60) | 0.1674 (60) |
| OpenAI | E | 0.9219 (55/55) | 0.3758 (55/55) | 0.8451 (60) |
| Anthropic | O | 0.1772 (179/60) | 13.3870 (179/60) | 0.1765 (180) |
| Anthropic | J | 0.1754 (180/60) | 13.5648 (180/60) | 0.1754 (180) |
| Anthropic | P | 0.1691 (60/60) | 0.0444 (60/60) | 0.1691 (60) |
| Anthropic | E | 0.9919 (46/46) | 0.0362 (46/46) | 0.7604 (60) |

平均は建築内で集約してから建築を等重みで平均した。表のvalid slots/buildingsは指標固有の分母であり、独立Nは最大60。IoUは高いほど良く、MAEは低いほど良い。材質・repair・追加/削除/置換の詳細と説明文provider×Builderの全組合せは `per_group.csv` に記録した。

global940は元HTTP401と予約を消さず、明示承認された認証再試行の成功を最終観測に選択した。global942のHTTP520は応答・usage・metrics=nullの運用欠測であり、モデル失敗やIoU=0へ変換していない。元401と補完520のunknown usage holdは2件とも保持する。会計の計算額を実請求額とは呼ばない。

事前指定20比較を3family（8/4/8）内でHolm補正した。p_holm<0.05はBuilder差2件、J−O差1件、oracle−O差6件。効果・方向・95%区間・p_raw/p_holm・共通valid pair/building数を全件掲載し、非有意差も省略しない。非有意は同等性の証明ではない。

JはOの情報保存型JSON再シリアライズであり、旧Structured IRではない。P/EはGT情報を追加する診断条件で、純粋な言語情報損失の因果効果とは解釈できない。RQ2は **NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES**。旧per-sample品質値は存在するが、現descriptionとの厳密な同一性を証明できず、一部は実際に内容が異なる。旧モデル平均・v5の流用や新規相関の計算はしていない。

HTTP520欠測のIoU [0,1]仮定は別のexploratory boundに限定し、p値を追加しない。限界は固定60建築、valid選択、モデル出力失敗、運用欠測1、品質スコア対応不足、oracleの解釈範囲である。追加APIなしの優先作業は、既存failure/metricの内訳確認、paired CIの精度確認、既存例の出典付き点検、新たに定義されたCPU品質評価を別の探索的証拠として整備することである。
