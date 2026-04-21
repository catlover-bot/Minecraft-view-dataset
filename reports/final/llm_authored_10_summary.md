# LLM-authored 10-case Diagnostic (Shared-source) Summary

- この結果は診断用セット（10件）で、Main/Supplementaryベンチとは別です。
- source author: `openai/gpt-5-mini`

## 1) OpenAI: direct vs structured
- direct IoU/F1/material/correct: 22.47% / 36.22% / 18.00% / 4.77%
- structured IoU/F1/material/correct: 20.49% / 33.47% / 30.49% / 7.71%
- delta(structured-direct): IoU -1.98pt, F1 -2.75pt, material 12.49pt, correct 2.95pt

## 2) Claude: direct vs structured
- direct IoU/F1/material/correct: 32.07% / 47.90% / 17.84% / 7.29%
- structured IoU/F1/material/correct: 29.23% / 44.37% / 34.65% / 11.84%
- delta(structured-direct): IoU -2.83pt, F1 -3.54pt, material 16.81pt, correct 4.55pt

## 3) Cross-provider (same shared-source 10 cases)
- direct OpenAI-Claude: IoU -9.60pt, F1 -11.68pt, material 0.16pt, correct -2.52pt
- structured OpenAI-Claude: IoU -8.74pt, F1 -10.90pt, material -4.16pt, correct -4.13pt

## 4) 説明品質
- OpenAI auto/strict/coarse/dim: 72.71% / 58.63% / 83.12% / 57.35%
- Claude auto/strict/coarse/dim: 66.97% / 37.62% / 59.29% / 83.47%

## 5) 注意点
- n=10 の診断セットなので、傾向は示唆的です（確定的主張は不可）。
- human study はプロトコルのみで、結果主張はしていません。
