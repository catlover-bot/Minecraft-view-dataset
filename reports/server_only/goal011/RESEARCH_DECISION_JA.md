# Goal011の研究判断

**結論は (A) 現在の診断結果を適切な最終主張とし、追加有料実験を行わない。** 質問A–Dの監査を完了し、ここでpost-hoc指標の追加を止める。既存60建築の結果は、再建築の失敗と評価指標に依存する順序を示す。image→textの情報損失、従来のcaption品質、可視性や建築的忠実度を識別する証拠にはならない。

baseline監査はPASS。Oのfilled-boxは保存説明のdimensions_estimateと固定原点だけ、Pは供給world_bboxだけから構築されていた。Builderとbaselineの同一input/GT、元occupancy shift、建築内平均→建築等重みを照合した。O/allの独立Nは各Builder60、適格ペアは177／179。aligned IoU差は −0.011960／−0.011615、Precision差は ＋0.092574／＋0.065285、Recall差は −0.211233／−0.153559、surface F1@1差は ＋0.099288／＋0.064520。既存探索的CIは変更せず、core-mask差異0、重大な数値矛盾0。既存2件の約10⁻¹⁷の丸め差を記録し、過去表は変更しなかった。

この符号組合せはO/allの全5サイズ層で維持されたが、説明providerをGeminiに限定するとIoU差は正になる。absolute Precisionは両Builderで負、shellやprovider別のsurface差も一様ではない。したがって、箱またはモデルの全面的優位ではなく、被覆・選択性・境界近傍距離の違いとして記述する。surfaceは空洞内壁を含む6近傍boundary voxel中心であり、撮影画像の可視面やmesh Chamferではない。[感度の読解](SENSITIVITY_REVIEW_JA.md)に不利な例も残した。

Eは101 valid出力すべてを独立検算した（OpenAI55、Anthropic46）。完全占有geometryは36/55・14/46、その中に材質不一致が22/36・4/14あった。19 truncationは別に維持し、CPU control60/60をモデル成功へ加算しない。034/OpenAIはC4912・M1794・不一致3118を再現し、一対一／many-to-oneのGT-informed上界でもM4752、残差160だった。出力上の系統的置換とラベル合流を示すが、内部認知機構は不明である。上界を予測の補正や新成功に扱わない。[全8例を含む読解](E_INTERPRETATION_JA.md)に限定を明記した。

rest30は固定評価workflowと非重複で、必要なOpenAI説明30件はある。一方、旧alignment解析とv5結果に全30の同ID参照があり、過去未評価・完全未見とは呼べない。固定1GiB探索上限による未検証ファイルも残る。[rest監査](REST30_REVIEW_JA.md)のreadinessはNOT_VERIFIEDである。

将来案は[未承認protocol](PROSPECTIVE_PROTOCOL_UNAPPROVED.md)の2候補だけに限定した。再検討する場合の第一候補はR1：rest30×2 Builders×OpenAI保存説明×r1＝60 calls、出力上限32768/回、入力予約18,000,000＋出力予約1,966,080＝19,966,080 tokens。主estimandはaligned occupancy IoUの建築等重み差、surface F1@1をcritical counter-metricとする。R3は180 calls・59,898,240予約tokensだが、反復内分散が未同定で追加120 callsの精度利得を保証できない。

eval60の変動が移るという仮定下で、R1のN30・IoUの計画半幅は約0.0181／0.0156、surface F1は約0.0350／0.0325。surfaceは計画許容幅0.03を超え、0.03自体も検証済みMCIDではない。これは将来CI・powerの保証ではなく、N30を十分と認定しない。価格・model availability・現行context適合は未検証で、費用は `REQUIRES_CURRENT_PRICE_VERIFICATION`。予約上限、実消費予測、invoiceを区別し、架空のUSD値を作らなかった。

この文書のcommitはpreregistration・課金承認ではない。既存991 calls／USD30枠は新しい承認ではなく、未知usage hold2件も維持する。将来別途実行するなら、現在価格と正確なmodel/schema対応を確認し、未来のoutcomeを見る前に設計・token/call/USD上限を再固定し、新たな実TTY人間承認を得る必要がある。自動retry0、未知usageでは予約を保持して停止、別のcheckpointと復元検証が必要。本Goalはauthority・ledger・approval・launcherを作らず、実行コマンドも作っていない。

image→textの情報損失を問うには、同じsceneと再構築課題に対し、既知の空間情報を保持する表現と特定情報だけを除去する表現、必要なら画像直接入力を、同じBuilder・schema・出力予算・実行時期・可能なsampling設定で制御したpaired介入が別途必要である。条件順序の乱択、反復変動、転記・容量失敗の分離も必要になる。Oは既にstructured JSONで、Jはtyped treeを保つ再シリアライズである。同じ保存Oからの反復生成だけではこの介入にならない。ここでは設計の必要性だけを述べ、実装・実行しない。

最も強い反対解釈は、現行60建築の結果を見て指標と説明を選んだ以上、rest30での新しい条件評価には依然として独立確認の価値がある、というものだ。Aはその価値がゼロという証明ではない。ただし、現在の失敗診断には新生成が不要で、R1を行っても棄却した因果主張や可視性問題は解消しない。過去曝露のsource chainと検証対象を明確にでき、独立確認が研究上必要で、要求精度と現行費用を満たす新しい証拠が整えば、Aを再検討してR1の新承認を求める余地がある。事前に固定された独立条件でトレードオフの符号が再現しないこと、または失敗率が大きく変わることは、現在の観測の一般化を反証し得る。

文献の網羅的な比較は実施しておらず、新規性やconference acceptanceを保証しない。以下のFAILは「主張をこの証拠から正当化できない」という判断であり、識別不能な科学的仮説を検定で偽と証明した意味ではない。

| claim type | 最終判定 | 許される範囲・残る不足 |
| --- | --- | --- |
| 1. 現cohortの指標依存baseline順序 | **PASS** | 60建築のpost-hoc記述。provider・対照・frame依存を併記し、一般的な優越性へ拡張しない。 |
| 2. Eでvalid条件付きの完全geometry生成とpalette caveat | **PASS** | 完全geometryの存在は36/55・14/46で確認。全valid完全・材質保証・全試行成功という主張は不可。 |
| 3. free-form languageがstructured JSONより空間情報を失う因果主張 | **FAIL** | OもJSON、Jはlossless reserialization。情報介入・生成変動・転記/容量を分離できずUNTESTABLE。 |
| 4. 従来caption品質RQ2対CPU GT-derived proxy | **INDETERMINATE** | 従来RQ2はNOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES。proxyは共有GT由来の属性診断としてのみ扱う。 |
| 5. rest30の独立評価readiness・費用・最小案 | **INDETERMINATE** | 必要入力と非重複は確認、旧解析曝露と探索上限・現行価格/モデル/長さが未解決。R1/R3ともUNAPPROVED、現在の判断はA。 |
| 6. visibility/occlusion・美的/機能的建築忠実度 | **INDETERMINATE** | 実投影・遮蔽・対応する妥当な品質尺度の証拠がなく、割合や原因を作らない。 |
