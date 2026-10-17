**Goal012：元のRQと、現在の証拠で識別できる範囲**

これは新しい実験・統計解析ではなく、Goal007–011の保存結果に対する研究監査である。参照時点はcommit `86a154b67a53d00de61f07fa006832204f7dec8a`。下記のsource ID、原本path、完全なSHA256、既存source bindingとの一致は [SOURCES.json](SOURCES.json) に記録した。値は既存表から引用し、再推定していない。

`EXACTLY_TESTED`は指定された操作的な問いに対応する測定が存在することを指し、仮説が真、一般化可能、因果識別済みという意味ではない。`PARTIALLY_TESTED`は狭い部分のみ測定、`NOT_TESTABLE`は必要な対比・対応が現在の資料にない、`INDETERMINATE`は関連する証拠があるが判定を確定できない状態である。

| 元の問い | 判定 | 実測範囲・独立N・根拠 | 識別していないもの |
| --- | --- | --- | --- |
| RQ1：保存画像説明から現行モデルがどこまで再構築できるか | **PARTIALLY_TESTED** | 固定60建築、保存2 Builder設定、3説明provider。O validは177/180・179/180 slots、各N60。限定した幾何・材質・信頼性測定はEXACTLY_TESTED。`g7_results`, `g7_denominators`, `g10v2_metrics`, `g11_claim_audit`。 | 現在の全モデル、他分布、可視外観・機能。情報欠落、Builder能力、sampling、容量の寄与。 |
| RQ2：独立した従来説明品質が、同一原文の再建築可能性を予測するか | **NOT_TESTABLE** | 元estimandの厳密対応集合・独立Nは未確立（null）。別のGT属性proxyは180説明／60建築、valid 177・179、両Builder valid平均176。`g7_rq2_audit`, `g8_definition`, `g8_source_map`, `g8_associations`。 | 従来品質尺度との対応、構成概念妥当性、独立予測性能。proxyの関連で代替しない。 |
| RQ3：同じ情報・能力条件のfree-form prose→明示的Structured IRで精度が変わるか | **NOT_TESTABLE** | 要求された対比の適格Nは未確立。実在する別対比J−Oは175/179ペア、各N60。`g8_semantics`, `g8_sampling`, `code_cohort`, `g7_comparisons`。 | prose対IRの意味構造差。表記・tokenizationと単発生成変動の分離。 |
| RQ4：image→text情報損失を能力・観測不能性・実行失敗から因果分離できるか | **NOT_TESTABLE** | Pは各60 valid、Eは55/46 valid、共通46建築。CPU control60/60。これらは情報追加の診断で、因果estimandのNではない。`code_partial`, `code_exact`, `g10_requests`, `g11_e`。 | 可視性、情報内容、出力長、転記、truncation、Builder/decodingを同時に変えない介入。 |

**RQ1の測定を支持する証拠と、一般化への反例。**

O/allの同一input・GT・common-validに対するBuilder−寸法filled-boxは、OpenAI／Anthropicの順に、aligned IoU −0.011960／−0.011615、precision ＋0.092574／＋0.065285、recall −0.211233／−0.153559、surface F1@1 ＋0.099288／＋0.064520。比較は177／179適格ペアを建築内平均し、60建築を等重みで扱う。Builder同士のO比較は別maskの176ペア／60建築である。保存CIは探索的で、新しい検定はない。[既存結果](../goal011/TRADEOFF_EXPLANATION_JA.md)、[感度と反例](../goal011/SENSITIVITY_REVIEW_JA.md)。

Gemini説明だけではaligned IoU差が＋0.027921／＋0.020939となる。O/all absolute precision差は−0.015776／−0.017812。shell、provider、frameで順序が変わる。「箱が常に優れる」「LLMが空間を理解できない」はこの観測から導けない。alignedはGTを使うoccupancy最適平行移動で、実world配置能力ではない。surfaceは空洞内壁を含む6近傍boundary voxel中心距離であり、画像の可視面ではない。[指標の原定義](../goal010v2/METRIC_SPEC.md)。

反対仮説は、順序の多くが占有量と測定対象の違いで説明でき、新たな空間認知機構を示さない、というもの。固定観測自体の反証条件はsource・mask・定義と値の不一致。将来への一般化仮説なら、事前固定した独立条件で符号や失敗傾向が再現しないことが反証になる。過去60の再集計は独立反証ではない。

**RQ2の欠損は、弱い相関ではなく対応証拠の欠損である。**

旧OpenAI score100行はfixed60のIDと重なるが、説明全文／GT hashがない。旧Anthropic100行もIDは重なるが、fixed60中40件で説明寸法に不一致がある。Geminiの対応score集合は見つかっていない。ID一致や3寸法一致だけで原文同一性を認めない。[厳密照合監査](../goal007/rq2_matched_scores_audit.json)。

Goal008 proxyは材質・寸法・typed field充足度からなるGT由来の別尺度で、summary等の全命題、独立した人間のcaption判断、従来caption metricではない。360探索は60建築を反復利用し、shared GT・建築難度・provider・valid選択の影響を受ける。建築内中心化でも因果交絡は消えない。[定義](../goal008/definition.md)、[統計定義](../goal008/stats_definition.md)、[探索結果](../goal008/EXPLORATORY_RQ2_SUMMARY.md)。

反対仮説は、正しく対応した独立品質尺度なら再建築可能性を十分予測できる、というもの。「品質は再建築を測らない」という仮説を反証する最小条件は、原文SHAと独立評価protocolを結んだ尺度が、建築単位の外部検証で事前に決めた有用な予測性能を示すこと。低い相関や非有意差だけでは無効性の証明にならない。必要な尺度・許容誤差・検証Nは未確定であり、ここでは評価を新規実行しない。

**RQ3では操作された表現が元の問いと違う。**

Oは最初からJSON説明レコードで、summary等の自然言語フィールドも持つ。Jはそのtyped treeを保持するlossless JSON再シリアライズで、旧Structured IRではない。P/EはGT属性／全離散仕様を追加するため、proseと同じ情報のIR側には流用できない。[表現監査](../goal008/audit/phase0_semantics.md)。

O/Jは各360予定slotsに対する別々の単発生成で、attemptはO361・J360。global940の承認retryは科学的反復ではない。Builder requestはtemperature、top-p、seedを明示固定していない。OpenAI応答metadataのtemperature=1・top-p=1を、全providerでの制御・決定性の証拠にしない。説明生成側のtemperature=0.2とも別である。[sampling監査](../goal008/audit/sampling_audit.json)、[provider交絡](../goal010/FACTOR_FINDINGS_JA.md)。

反対仮説は、情報等価なprose/IRに有用な差はなく、J−O差は単発変動等による、というもの。反証可能なIR改善仮説には、内容等価性を独立検証したpaired表現、同じBuilder/schema/cap、可能なdecoding設定、順序制御、反復が必要。有用差を排除する精度のある差、またはprose優位が改善仮説を反証し得る。非有意だけで同等と判定しない。

**RQ4に関係する強い観測は、下流失敗が実在することまでである。**

E valid101の完全occupancyは36/55・14/46、そのうち材質不一致22/36・4/14。truncation19（5/14）は別に保持する。034/OpenAIはg=p=C4912、M1794、不一致3118。GT-informed最適対応上界でもM4752、残差160で、純粋な可逆swapではない。これは出力パターンの証拠であり、内部認知機構・補正済み成功・image→text損失の割合ではない。CPU60/60は表現とexecutorのcontrolだけを検証する。[E診断](../goal011/E_SYMBOL_BINDING_FINDINGS_JA.md)、[全例の解釈](../goal011/E_INTERPRETATION_JA.md)。

反対仮説は、Oの低精度が不可視形状やBuilderの転記・容量に支配され、自然言語化が主因ではない、というもの。特定の観測可能factだけを除去／回復するpaired介入と、同じ出力課題の能力・容量controlが必要。十分な操作確認と精度の下でfact回復が効かず、下流controlの改善で差が消えるなら「そのfact欠落が主要因」という限定仮説への反証になる。O反復をrest30で増やしても、この因果識別は成立しない。

**母数・独立性の保留。** 960 slots、934 valid、25モデル出力失敗、HTTP520欠測1、961 fixed-scope attempts、歴史的991 calls、unknown-usage予約2件を維持した。欠測を0へ補完せず、正常empty成功も再分類しない。rest30はeval60/smoke10と非重複だが、全30に旧解析・v5参照があり、探索上限も残る。独立検証readinessはINDETERMINATE／既存実行判定NOT_VERIFIEDである。[rest原監査](../goal011/REST30_REVIEW_JA.md)。
