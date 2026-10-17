**Goal013：独立scoreの有効性判定**

`POST_HOC_EXPLORATORY / GT_INDEPENDENT_SUMMARY_ALIGNMENT`。

**source bindingはPASSだが、score-validは `NOT_ASSESSABLE_NO_SCORER`。** 検証済みのoffline採点器がなく、実summary得点は0/180、実image–text pair採点は0件である。CLIPScore候補も標準cosine候補も未採用で、active score ID、checkpoint、実weight/tokenizer/scorer SHAは `null`。`COMPUTATION_NOT_PERFORMED` を維持する。[採点器監査](OFFLINE_SCORER_AUDIT.json)、[固定規約](SCORING_PROTOCOL.md)。

| source検証項目 | 保存された結果 |
| --- | --- |
| O説明record／建築 | 180／60、各provider 60 |
| request指定画像 | 各6視点、unique image paths 360 |
| image–text pair計画上限 | 1080。採点済み件数ではない |
| source inventory | 796ファイル、前後不変 |
| summary JSON型 | 180件すべて文字列 |
| 非空summary | 178件 |
| 空summary | Geminiのbuilding_054・building_061、各0文字・0 bytes |

空summary 2件は `MISSING_EMPTY_SUMMARY` として行を保持する。得点0への補完や他フィールドによる代用はしない。非空178件も採点成功・token上限内という意味ではない。実tokenizerがないため、有効token長、切捨て量・割合、token代表性gate、6視点得点のmin/max/population SDはすべて `null / NOT_COMPUTED`。

保存requestは同一建築の3 providersで順序付き6画像bytesとpromptが一致した。ただし、requestが指定した画像の照合であり、transportによる実送信、モデルによる視認、画像内の可視性・遮蔽を独立に証明していない。12候補画像とrequest指定6画像も区別する。

将来のscorer入力allowlistは `summary` と `images_ordered` のみ。Phase 0 receiptにはGT grid/bbox・Builder outcomeファイルの新規open 0、payloadへのGT/outcome混入なしが記録されている。full manifestの既往GT provenanceはO原文の同一性監査に限り用い、採点payloadへ渡していない。[fixture検証](VALIDATION.json)のsource hash・payload境界PASSはこの制約の検証であり、未実行の実scoreがGT非依存に計算されたという実証ではない。実画像でのreplay・平均不変性・model決定性も `NOT_VERIFIED_NO_SCORER`。

採点対象はsummaryだけであり、O JSON全文や人間の総合caption品質ではない。モデル出典・重み・tokenizer・前処理・context・依存を束縛できる採点系が揃うまでは尺度の有効性を判断できず、旧RQ2の判定を変更しない。

出典：`/root/minecraft-runs/goal013-20261017T111808Z/phase0/SOURCE_BINDING_RECEIPT.json` と `SOURCE_BINDING_180.json`。本ノートは保存証跡の読解のみで、推論・新しい統計・旧outcome再読を行っていない。
