# llm_authored_10 Minecraft Grounded Summary

このレポートは `llm_authored_10` 診断セットの **Minecraft実画像入力版** です。
メイン200件ベンチ（Main/Supplementary/Execution-gap）とは混ぜません。

## 実行条件
- OpenAI dataset: `/Users/hirotaka-m/Minecraft-view-dataset/datasets/llm_authored_10_minecraft`
- Claude dataset: `/Users/hirotaka-m/Minecraft-view-dataset/datasets/llm_authored_10_minecraft_claude`
- source image origin: `minecraft_capture`
- source build origin: `minecraft_instantiated`
- direct rebuild image origin: `minecraft_capture`
- direct rebuild build origin: `minecraft_instantiated`
- structured rebuild image origin: `minecraft_capture`
- structured rebuild build origin: `minecraft_instantiated`

## Provenance検証
- OpenAI valid cases (source+direct+structured all Minecraft grounded): 10/10
- Claude valid cases (source+direct+structured all Minecraft grounded): 10/10

## OpenAI (gpt-5-mini)
- Description: auto 80.62%, strict material F1 72.52%, coarse material F1 88.95%, dimension 66.50%
- Direct rebuild: IoU 29.16%, F1 44.56%, material 23.79%, correct placement 13.24%, repair edit 1.335
- Structured rebuild: IoU 32.69%, F1 48.82%, material 51.40%, correct placement 33.08%, repair edit 1.126
- Structured - Direct: IoU +3.53 pt, F1 +4.26 pt, material +27.61 pt, correct placement +19.84 pt

## Claude (claude-haiku-4-5)
- Description: auto 75.61%, strict material F1 60.82%, coarse material F1 77.53%, dimension 71.93%
- Direct rebuild: IoU 25.56%, F1 39.88%, material 11.06%, correct placement 6.36%, repair edit 1.462
- Structured rebuild: IoU 27.86%, F1 42.45%, material 44.19%, correct placement 25.62%, repair edit 1.301
- Structured - Direct: IoU +2.30 pt, F1 +2.58 pt, material +33.13 pt, correct placement +19.26 pt

## OpenAI vs Claude（同一 shared-source）
- Direct: OpenAI-claude差 IoU +3.61 pt, F1 +4.69 pt
- Structured: OpenAI-claude差 IoU +4.83 pt, F1 +6.37 pt

## 以前の ambiguous/synthetic-run との差
- OpenAI structured IoU: 20.49% -> 32.69% (+12.20 pt)
- Claude structured IoU: 29.23% -> 27.86% (-1.37 pt)

## 注意
- これは10件の診断実験で、統計的確定ではなく傾向確認です。
- 人手実験の結果は含みません（プロトコルのみ）。
