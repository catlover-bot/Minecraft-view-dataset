新規API上限USD5は**未承認の提案**であり、80件完了の保証ではない。研究API送信0、実請求額は未発生／未照会。

2026-10-17 17:15–17:18 UTCにこのサーバーで取得した公式資料のstandard料金は、GPT5mini input0.25/output2、Haiku4.5 input1/output5 USD/1M tokens。PRICING_EVIDENCE.jsonにURL/実取得UTC/byte SHAと確認内容を保存した。cache割引、Batch、Priority、外部tool、cache writeを予算削減の前提にしない。OpenAIのoutput usageにはreasoningを含める。

| 予定40calls/Builder | 入力本文UTF8合計bytes | 保存payload合計bytes | HTTP JSON body合計bytes | body最小–最大bytes |
| --- | ---: | ---: | ---: | ---: |
| OpenAI | 70,702 | 235,090 | 240,770 | 4,494–8,013 |
| Anthropic | 70,702 | 233,970 | 239,530 | 4,463–7,982 |

40固有payloadを2反復で使用するため表は送信予定80行を数える。schema/prompt/framingを含む保存ファイルと実transportのJSON serializationはサイズが異なる。COST_ESTIMATE.jsonに全80行のサイズとwire body SHAを保存した。既存実行環境に対応tokenizer（tiktoken）はなく、実token数はNOT_AVAILABLE。CLIP tokenizerをBuilder計数へ流用せず、bytesをtokensと呼ばない。新install/token-count APIは0。

過去991試行を調べ、同Builder・保存OpenAI説明・O条件の全129該当試行のうち128件がknown usage（各Builder64）、OpenAI401の1件はunknownのまま除外した。成功だけでなく失敗・旧retryも含め、raw usageと台帳を全128件照合した。全known平均×40の参考値はOpenAI **USD0.655729219**、Anthropic **USD0.442420000**、合計 **USD1.098149219**。元fixed961に限定した各60knownの参考合計はUSD1.097091834。前者には初期30試行の該当8件も含み、うち4件は旧prompt・cap16384である。全件表・区分・source hashesはHISTORICAL_USAGE_REFERENCE.jsonに保存した。

旧Oは管理metadataと原serializationを含み、新S/Fのinput/output長は未知である。上記は参考外挿であり予測保証ではない。選択10件や有利な成功例だけへ絞っていない。

| Builder | 保守的input予約tokens | output cap | 1request最大USD | S/F block最大USD | 全40最大USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenAI | 400,000 | 32,768 | 0.165536 | 0.331072 | 6.621440 |
| Anthropic | 200,000 | 32,768 | 0.363840 | 0.727680 | 14.553600 |

既存の全documented contextをinput予約に用い、別途output上限を加える方式を維持する。schema/framingを含む安全側の予約で、実token数の認定ではなく、実際にinputとoutputの両上限が同時達成できるとの主張でもない。全80最大は **USD21.175040**、USD5との差は **USD16.175040**。最悪条件で全80を覆いたい場合の推奨提案額は端数切上げ **USD21.18**だが、今回のplanはUSD5のままで、上限を変更／承認していない。

将来は全80を永久予約せず、各隣接paired blockの保守的額を先に確保する。known精算＋unknown/unsettled holdsを上限と比較し、次blockを確保できなければ送信前停止する。外部障害で片側だけ送信された場合は未完ペアを残す。usage不明を0円にせず、再送もしない。24時間を超えた価格証拠は同planへ独立追記して更新し、価格／モデル／契約変更は新承認を要する。

旧予算計上USD21.56083625（Goal007 READMEと991台帳の再集計）には2unknown holdsを含む。これは請求額ではなく、旧USD30の残額を新scopeへ流用しない。仮に新提案USD5を使い切った場合の参考合算USD26.56083625も、請求額／承認済み予算ではない。
