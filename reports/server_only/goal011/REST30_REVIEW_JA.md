# rest30：未評価workflowと過去利用の区別

POST_HOC_EXPLORATORY / CPU_DIAGNOSTIC。`REST30_FEASIBILITY.json` と `REST30_SOURCES.csv` の保存結果を読解した補足で、探索上限を拡張していない。

元のrestは30 unique、5サイズ層に各6件。eval60・smoke10との重複は0で、確認済み4固定planにもrestは出現しなかった。したがって、Goal005–010のこの固定評価workflowでは未評価である。

GT、bbox、metadata、保存候補画像360 unique paths（各12枚）、3 provider×30の説明90件をsource/hashへ結び付けた。90件の保存requestはすべて6枚で、meta index順は `[0,2,4,6,10,11]`。これは保存requestの画像対応であり、独立したtransport記録による送信証明や、visibility/occlusionの検証ではない。将来候補のOpenAI説明は30/30利用可能だが、正確なprovider token数・将来出力の完了可能性はNOT_VERIFIEDである。

**完全未見・過去未評価という扱いはできない。** 以下の旧alignment表には全30 scene IDのexact-v4参照がある。

| source path | SHA256 |
| --- | --- |
| `/root/minecraft-runs/goal002-20261013-141854/alignment_comparison.csv` | `8d2ce5cdc57070ff73cc197feda0fe4595eb770b9b1e260190e67ff9310ebb71` |
| `/root/minecraft-runs/goal002-20261013-141854/final_cpu/legacy/alignment_comparison.csv` | `d7d2943a1b78da0fc8c94b59bf53d85911957729f29a8055383a733a1430cbe9` |
| `/root/minecraft-runs/goal002-20261013-141854/verified_cpu/legacy/alignment_comparison.csv` | `8d2ce5cdc57070ff73cc197feda0fe4595eb770b9b1e260190e67ff9310ebb71` |

v5名を含むlegacy geometry結果も、全30に対応する540 unique paths（3 snapshot×180）がhash済みだった。例は `.../final_cpu/legacy/legacy_geometry/buildings_100_v4_building_000/rebuild_world_schema_material_v5_repair_openai_gpt_5_mini.json`、SHA256 `66946ea78b7982caf4585d688b20120528e299040b281307ee330f0f588d3afa`。これは過去の解析利用の正の証拠だが、元Builder completionの成功・GT/inputとの完全なsource chainを再検証した証明ではない。snapshot複製を独立した実験回数と数えない。

探索は27,278候補中12,730ファイルをhashし、1,073,741,792 bytesで固定の1GiB上限に達した。7,404候補は `SEARCH_RESOURCE_LIMIT`、7,144候補は `BINARY_CONTENT_NOT_SEARCHED`。rest30全30に対応する旧v5のplan/actions/voxels 540候補も前者に含まれる。`exact_v4_prior_builder_artifact_references=0` は検出済み分類の値であり、「過去Builder出力が存在しない」証明ではない。未探索理由とpathはJSONにすべて保持した。

人手関連は132参照／86 paths、そのうちexact-v4は18参照／14 pathsで `building_012/076/086` の3建築。`outputs/human_image_rebuild/case_packages/manifest.json` のSHA256は `24e13f10ec59cb9fc5a3e7aeb15e61a8c581ccd0946483b06e1f57c22a3edf1d`。case packageの存在は人手研究の実施完了や評点の証明ではない。figure/renderは252 paths・14同名建築への参照があったがexact-v4は0で、別dataset版やbasenameだけの曖昧さを残した。

**readinessはNOT_VERIFIED。** 新しいO条件でのholdout-condition評価候補にはなり得るが、過去の解析曝露、出力source chain、探索外履歴を解消せずに独立性を強く主張できない。新分布でも完全未見でもない。現在の結論は追加有料実験なし（A）とし、将来の候補設計は未承認のまま保存する。
