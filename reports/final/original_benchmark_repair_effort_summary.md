# Original Benchmark Repair-Effort Summary

この結果は `datasets/buildings_100_v1` / `datasets/buildings_100_v4` に対する追加診断です。
Main/Supplementary の既存IoU/F1等は変更せず、repair-effort軸を追加しています。

## Coverage
- models: claude, openai
- regimes: main, supplementary
- datasets: v1, v4
- conditions: 8, cases: 600

## Main (shared hparams)
- claude all_200: IoU 20.55% / F1 33.45% / material 20.66% / correct 8.29% / edit 1.492 (add 0.682, del 0.557, rep 0.253)
- claude v1: IoU 24.29% / F1 38.48% / material 17.79% / correct 7.97% / edit 1.484 (add 0.631, del 0.552, rep 0.301)
- claude v4: IoU 16.81% / F1 28.42% / material 23.53% / correct 8.62% / edit 1.500 (add 0.732, del 0.562, rep 0.206)
- openai all_200: IoU 23.75% / F1 37.46% / material 29.78% / correct 13.18% / edit 1.595 (add 0.592, del 0.709, rep 0.294)
- openai v1: IoU 29.44% / F1 44.75% / material 28.96% / correct 13.96% / edit 1.772 (add 0.461, del 0.911, rep 0.401)
- openai v4: IoU 18.06% / F1 30.16% / material 30.60% / correct 12.39% / edit 1.417 (add 0.723, del 0.507, rep 0.188)

## Supplementary (model-tuned)
- claude all_200: IoU 0.00% / F1 0.00% / material 0.00% / correct 0.00% / edit 0.000 (add 0.000, del 0.000, rep 0.000)
- claude v1: IoU 25.89% / F1 40.45% / material 19.62% / correct 8.41% / edit 0.000 (add 0.000, del 0.000, rep 0.000)
- claude v4: IoU 18.67% / F1 31.04% / material 23.48% / correct 8.39% / edit 0.000 (add 0.000, del 0.000, rep 0.000)
- openai all_200: IoU 25.37% / F1 39.68% / material 25.99% / correct 10.31% / edit 1.956 (add 0.475, del 1.086, rep 0.395)
- openai v1: IoU 30.30% / F1 45.88% / material 22.47% / correct 10.14% / edit 2.245 (add 0.321, del 1.387, rep 0.537)
- openai v4: IoU 20.45% / F1 33.48% / material 29.52% / correct 10.47% / edit 1.666 (add 0.629, del 0.784, rep 0.253)

## Near-Miss (low IoU but low edit)
- criterion: IoU < 0.20 and edit_distance_over_gt <= 0.50
- matched cases: 0

## Interpretation guardrails
- repair-effortは IoU/F1 の置き換えではなく補助指標です。
- Main と Supplementary は混ぜずに解釈してください。
- これは original benchmark 追加診断で、llm_authored_10 とは分離しています。
