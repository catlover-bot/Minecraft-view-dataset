# E条件：固定geometry上の材質記号対応の診断

POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC

元のE出力120件を全件保持し、成功101件と打切り失敗19件を分けた。成功のうち新診断を計算できたのは101件。GT→固定cuboid→参照操作のCPU完全一致controlは60建築で、モデル成功へ加算しない。

既存absoluteと保存済みoccupancy最適shiftのalignedを使用した。新しいalignmentやalias置換は行わず、元操作・voxel・評価値を変更していない。

| Builder | frame | 成功 / 診断N | 元M/C | 単射上界M/C | 多対一上界M/C | 元M/g | 単射上界M/g |
| --- | --- | --- | --- | --- | --- | --- | --- |
| openai | absolute | 55 / 55 | 0.933593 | 0.974694 | 0.983818 | 0.861242 | 0.898980 |
| openai | aligned | 55 / 55 | 0.933593 | 0.974694 | 0.983818 | 0.861242 | 0.898980 |
| anthropic | absolute | 46 / 46 | 0.867522 | 0.970875 | 0.974070 | 0.861291 | 0.963454 |
| anthropic | aligned | 46 / 46 | 0.867522 | 0.970875 | 0.974070 | 0.861291 | 0.963454 |

固定focusの保存済み原本に対する独立countsと上界:
- building_034 / anthropic: g=4912, p=4912, C=4912, M=4912; 元M/C=1.000000, 元M/g=1.000000, 単射上界M=4912, 多対一上界M=4912。
- building_034 / openai: g=4912, p=4912, C=4912, M=1794; 元M/C=0.365228, 元M/g=0.365228, 単射上界M=4752, 多対一上界M=4752。

表は各Builderの成功建築を等重みで平均した値である。M/Cは共有占有内の厳密材質一致、M/gはGT全占有を分母とする。C=0ではM/CはNULLであり0補完しない。voxelを全件合算したΣM/ΣC・ΣM/Σgは別欄に保存し、この等重み平均と混同しない。

対応上界は各予測ごとにGTを見て最大化した診断値である。単射は予測ラベルをGTラベルへ部分一対一対応させ、多対一はこの制約を緩める。geometryの欠落・余剰は変えられない。上界への改善は記号対応だけで説明し得る量であり、モデル内部の転記機構や原因の証明、修正済み出力、新しい成功例、学習した一般化対応ではない。最適対応の辞書順witnessと全最適解で可能な各ラベルの相手を保存し、同率・未観測ラベルを明示した。

参照cuboidとの操作照合は絶対world XYZのinclusive boundsを使用する。参照・出力の両方で一意かつ非重複の場合だけ材質一致/不一致を分類する。重複・上書き・分割・併合・ずれ・carve・対応不明は曖昧のまま残す。別分割が同じ最終geometryを作れるため、境界不一致を欠落や意図的shiftとは断定しない。

保存raw responseのaction本文を凍結parser/schemaで再検証し、parsed操作と一致することを確認した。材質IDは凍結executorの規約に従う（任意minecraft:接頭辞を除去、air類は欠損）。rawのblock綴りとcanonical IDを分けて保存し、stonebrick/stone_bricks等の同義語補正はしない。公開JSONでは操作座標を複製せず、一意対応のlabel pair件数・全操作のraw/canonical ID頻度・source hashを保存する。

building_034/OpenAI/Eは結果によらず固定focusとして監査JSONへ収録した。全101成功の表・全confusion・両上界を保持し、良い対応が得られた例だけを選択しない。

出力上限は元plan、token数は保存済みprovider usageのみ。参照operation数・UTF-8 byte数をtokenへ換算せず、未完了出力の正しさや上限内完了を推測しない。19件の打切りはgeometry/material診断の分母から除き、失敗として別記録に保持した。

成功例条件付きの診断であり、両Builderの成功集合差を無視した因果比較は行わない。研究API・モデル・token-count・キー照会は0件。
