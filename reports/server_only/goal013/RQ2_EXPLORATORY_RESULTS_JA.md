**Goal013：RQ2探索的関連は未計算**

`POST_HOC_EXPLORATORY / GT_INDEPENDENT_SUMMARY_ALIGNMENT`。

**結果は `COMPUTATION_NOT_PERFORMED`。** source bindingの180説明／60建築は採点入力の計画規模であり、相関解析のactual Nではない。検証済みoffline scorerが0件のため、得点固定とoutcome結合へ進んでいない。採点成功0/180、image–text pair採点0/1080計画上限。paired denominatorと関連解析のactual Nは、未形成のため `null` とする。

| 規定した探索 | 推定値・CI・paired denominator・actual N |
| --- | --- |
| OpenAI BuilderのO IoUとのpooled Spearman | `null / NOT_COMPUTED` |
| Anthropic BuilderのO IoUとのpooled Spearman | `null / NOT_COMPUTED` |
| both-valid同一説明の2 Builder平均IoUとのSpearman | `null / NOT_COMPUTED` |
| 各Builderの建築内pairwise concordance→建築等重み | `null / NOT_COMPUTED` |
| identified completed failureだけのfailure0感度 | `null / NOT_COMPUTED` |
| provider別推定可能性、score–token長関連、view得点分布 | `null / NOT_COMPUTED` |
| planned60 cluster／2000 bootstrap／seed20261017 | 未実行 |

無相関、負相関、順位不一致、効果0という結果は得ていない。旧v5得点、Description Overall、Goal008のGT proxy、Builder欠測の0補完による代用はしていない。Gemini building_054・061の空summaryは欠測であり、caption品質0とは評価していない。p値・有意性判定・追加の統計もない。[採点規約](SCORING_PROTOCOL.md)と[有効性判定](SCORE_VALIDITY_JA.md)を参照する。

将来の関連にも残る交絡を、実bindingに保存されたmetadataから再掲する。次表の日時はdescriptionの保存UTC日であり、独立に確認されたAPI送信時刻ではない。

| description provider | 保存model ID | 保存UTC日 | recorded max_tokens |
| --- | --- | --- | ---: |
| openai | `gpt-5-mini` | 2026-02-28 | 1800 |
| anthropic | `claude-haiku-4-5-20251001` | 2026-03-01 | 1800 |
| gemini | `gemini-3.1-pro-preview` | 2026-04-18 | 4096 |

これは保存model ID/version文字列の確認であり、`gpt-5-mini` やGemini previewが実際に使ったbackend snapshotを独立に確定したものではない。旧モデルの現行availability照会も行っていない。

全providerの保存temperatureは0.2、top_p/top_k/seedは `null`。recorded llm_seedはOpenAI/Anthropicが `null`、Geminiが `-1` で、これを固定乱数seedの証明とは扱わない。これらは保存requestの記載であり、provider側で同じ意味のsampling条件が適用された保証ではない。

prompt SHAは `940e5d241e90bbdccd90b0d67dea1deec23d4bcb324c18d86098eecf5dec4d0c`、system prompt SHAは `a36506a019c9d208b85eb43537584079134493de614f91d4ac6b5310c58ab076` で一致する。しかし同じ画像・promptでも、モデル版、生成日、1800対4096の出力予算、samplingは交絡したままである。採点モデルのtoken contextはこのdescription生成予算とは別で、現在は未選択・未測定。

原RQ2の「従来の独立説明品質が再建築可能性を予測するか」は未解決である。将来summary-image関連度が測れても、O全文の品質や人間の品質判断、因果的な情報損失、外部分布への一般化を直接検証したことにはならない。

出典：`/root/minecraft-runs/goal013-20261017T111808Z/phase0/SOURCE_BINDING_180.json` と同directoryのreceipt。上記metadataの読取り以外にoutcomeを再読・再集計していない。
