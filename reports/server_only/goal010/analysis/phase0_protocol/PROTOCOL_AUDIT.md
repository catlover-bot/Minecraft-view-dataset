# Goal010 プロトコル監査（CPU_DIAGNOSTIC）

180 request記録と実参照画像を全数照合した。入力画像枚数分布は {'6': 180}、保存候補view数分布は {'12': 180}。meta.viewsは保存候補であり、送信入力はrequest.imagesで特定した。同一sceneの3 providerで画像bytes・順序の一致=True、prompt hashes一致=True。

- openai: models=['gpt-5-mini'], temperature=[0.2], max_tokens=[1800]; recorded UTC 2026-02-28T15:03:32.530792+00:00–2026-02-28T15:29:34.888485+00:00
- anthropic: models=['claude-haiku-4-5-20251001'], temperature=[0.2], max_tokens=[1800]; recorded UTC 2026-03-01T04:49:59.031522+00:00–2026-03-01T05:04:35.457764+00:00
- gemini: models=['gemini-3.1-pro-preview'], temperature=[0.2], max_tokens=[4096]; recorded UTC 2026-04-18T15:35:51.120649+00:00–2026-04-18T19:44:27.411572+00:00

created_atは保存記録のUTCで、実API送信開始・終了時刻は未記録。producerは応答取得後にdescription/request記録を保存する。記述モデルversion、max_tokens、実施時期等の設定込みの観測であり、モデル固有能力の因果比較を支持しない。

100建築をGT occupied数とscene IDで順位付けし、5層（各20建築）内の固定乱択で評価12・smoke2・残り6を選んだ。seed=20261013、総数60/10/30の再生が一致。独立評価単位は60建築であり、180記述・360Builder観測ではない。

Oは保存JSON説明レコード、Jは同じtyped JSON treeの再シリアライズである。P/Eの実入力形式・Builder sampling・strict evaluatorはrootの共通source auditと固定manifest/codeに束縛する。meta.generation.origin、storage bbox、camera positionsは別概念として保存し、occupied bboxや光学中心の代用にしない。GTの読み込みは既存YXZ→XYZ readerを使用する。

保存camera yaw/pitch/FOVはplanned/requested metadataであり、実際の投影を独立検証した証拠ではない。actual XYZがあってもactual yaw/pitch、カメラ光学中心・eye offset、effective projection、深度/遮蔽の画像対応を保証しない。可視性判断はPhase4へ分離し、未検証なら割合を作らない。

元request・prompt・response本文やbase64、credentialsは出力していない。出典path/hash、設定のallowlist、画像の実寸法、カメラ記録だけを保存。研究API送信・キー照会・token-count APIは0。
