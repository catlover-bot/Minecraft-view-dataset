Goal015B：非空再建築0件の原因切分け

**48件の空グリッドの直接原因は、保存rawにある操作列が全件`carve`（削除）だけだったこと。空配列は0件、配置後の全削除も0件だった。** 最初から空のグリッドに削除だけを適用しても建築はできない。独立抽出・独立実行は既存評価と一致し、既存評価の訂正は不要だった。なぜモデルがこの操作列を返したかは未確定。追加研究API送信0、残8件は未承認・未送信のまま保留した。

| 48 emptyの分類 | 実件数 | 根拠 |
| --- | ---: | --- |
| rawの空operations配列 | 0 | すべて2–504操作の非空配列 |
| 削除だけを空の初期グリッドへ適用 | 48 | `carve`計2,068、`set`/`fill`各0 |
| 配置後に全部削除 | 0 | 非air書込み0、最大occupied数0 |
| 抽出・実行の不具合 | 0 | 72 text一致、48予測の全座標・材質一致 |
| その他・不明なempty分類 | 0 | 全48件を上記へ分類済み |

削除訪問数は計4,773,088（重複訪問を含む）。実際に存在したvoxelを削除した数は0。最初の配置indexと非空→空の遷移indexはすべてNULLであり、「配置されていた建築が消えた」とは解釈しない。protocol-validは構文・実行規約への適合であり、建築再現成功を意味しない。

**確認した境界。** 親planの80行とledger/evaluationのcall identityを対応させ、descriptorのSHA256・bytesを照合した。標準JSON decoderによる独立読取りは重複キーを拒否し、API配列順でtextを連結する。OpenAIは全36件でreasoning項目1個とmessageのoutput_text 1個、Anthropicは全36件でtext 1個。reasoning項目は非textとして記録し、その内容を採点や公開に用いていない。refusal、多重text、text欠落、未知のcontent型は実対象では0。保存textとのbyte一致は72/72、response IDとraw SHAは各72種類、text SHAは71種類だった。同一textのcall027/054もraw・response IDは異なる。同一本文だけでfixture混入とは判定しない。

全payloadは凍結prompt/schema/input構築規則に一致した。OpenAI rawのechoはprompt、schema tree、format契約、出力capについて36/36一致。format全体にはAPI側の`description: null`追加があり、formatのbyte/tree全体一致とは区別した。Anthropic rawはinput/prompt/schema全体を返していない。保存payloadの存在とローカル実行記録は確認できたが、独立したremote request-body記録はなく、実transportへ渡った全bytesの外部証明まではできない。

保存launcher logは実planの通常CLI、`fixture_only:false`、72件終了後の予算SAFE_STOPを示す。凍結CLIはtest hookを公開せず、runtimeは実planへのadapter/confirmation/checkpoint注入を拒否する。非canonical raw path、fixture/mock response ID、ledgerとの不一致は見つからなかった。これらは限定されたローカル来歴の証拠であり、モデル内部や外部サービスの原因を断定するものではない。

**独立Executorと正例。** 新しい`independent.py`は旧validator/executeを内部で呼ばず、有限XYZの三重ループと疎辞書でinclusive endpoints、逆順端点、last-write-wins、air/carve削除、元の座標・操作・訪問・live voxel上限を再現した。指定8 fixture、境界拒否、材質変更、決定的な小規模混合例を検証した。12個の非空正例（両provider）は実Goal015の抽出→validate→execute→evaluateを通過した。全48実応答を独立実行した結果、保存predictionの座標・材質・件数と完全一致した。GTへの切詰めや補正は行っていない。

旧Oの比較は同じbuilding/説明生成元/Builderの非空成功からID順先頭だけを選んだ。両Builderともbuilding003/OpenAI記述となり、OpenAIは52操作・1,856voxel、Anthropicは57操作・1,817voxelだった。両方とも現在の評価経路で非空を保持した。詳細・入力差・原requestの証明範囲は [INPUT_DELTA_AND_REPLAY.md](INPUT_DELTA_AND_REPLAY.md)。自己参照GTによるCPU正例であり、新しい研究outcomeではない。

**失敗24件。** truncation22件はすべてAnthropic。rawの`stop_reason=max_tokens`、output usage=32,768を全件で確認した。本文は49,913–60,502 bytes、22件ともJSON未完了。保存prefix中で完全に読めた18,550操作はすべてcarveで、set/fillの過剰展開は観測されなかった。17件に同一操作の再出現があり、同じ操作の最大連続数は862。これらはprefix観察であり、JSONを閉じ直したり途中操作を実行して成功扱いしたりしていない。内部の反復原因は不明。Anthropicのreasoning token数はrawにないためNULLのまま。

validation2件は材質・key・型の誤りではなく、元の累積voxel touch上限2,000,000超過だった。call019（Anthropic/building092/S/r1）はoperation index111、call059（OpenAI/building024/S/r1）はindex99で拒否された（0始まり）。どちらも全操作carveで、completed/end_turnで返った有効JSONだった。上限を緩和せずNULLを維持した。具体的な拒否文と前後の訪問数は [TRUNCATION_VALIDATION_AUDIT.csv](TRUNCATION_VALIDATION_AUDIT.csv)。

| Builder | 送信済み | empty | 非empty | truncation | validation | 親の既知計上USD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI | 36 | 35 | 0 | 0 | 1 | 0.477346 |
| Anthropic | 36 | 13 | 0 | 22 | 1 | 3.960311 |
| 合計 | 72 | 48 | 0 | 22 | 2 | 4.437657 |

raw usageと凍結単価をDecimalで再計算し、callごとの既知費用と親合計USD4.43765700に一致した。元のcache割引を使わない計上規則であり、新たな請求書ではない。OpenAI output233,436 tokensのうちrawで確認できたreasoningは196,608 tokensで、outputに既に含まれるため二重加算しない。Anthropic outputは778,753 tokensで、上限到達22件のoutput部分だけで720,896 tokens／USD3.604480を占める。旧80件外挿USD1.098149219に対し今回72件で約4.041倍だが、同じ件数・条件の比較ではなく、差の全てをreasoningへ帰属させない。8セルと9建築の全内訳は [BY_CELL_8.csv](BY_CELL_8.csv)、[BY_BUILDING_9.csv](BY_BUILDING_9.csv)。

**S/Fについて言える範囲。** 保存済みのIoU/surface F1は正常emptyも既定failure0も0に退化している。既定completed-attempt utilityは各Builder18 paired blocks/9建築、common-validはOpenAI17 blocks/9建築、Anthropic2 blocks/1建築だった。後者のAnthropic CIはNULL。S/F差0やCI[0,0]を全文情報の無効性・同等性・情報損失の証明としない。新しい有意性判定、equivalence検定、統計量の追加計算はしていない。[RESEARCH_DECISION_JA.md](RESEARCH_DECISION_JA.md)に判断を固定した。

**保全と保存。** 新runは `/root/minecraft-runs/goal015b-20261020T090529Z-5f01c85f`。親plan、ledger/approval/authority、72 raw/text、payload、評価、凍結コード55件、子plan/制御と今回選んだ旧O原本だけを対象とした。親上限USD5、SAFE_STOP、子未承認、残8件NOT_SENT、不明usage0を保持。元料金証拠の期限切れは記録だけで更新していない。旧Draft417384305/asset656494745の297ファイル復元記録を再利用し、再download/uploadやRelease操作は0。CLIP・全過去実験・数万ファイルの再監査はしなかった。

今回の保存対象は新しい診断コード・テスト・小さい表・報告だけ。raw全文、原summary、実voxel座標配列、重み、環境、秘密情報は追加Git公開していない。最終のhash保全、限定テスト、資源実測、全要件照合は [COMPLETION_AUDIT.json](COMPLETION_AUDIT.json)、成果物対応は [MANIFEST.json](MANIFEST.json)を参照。通常commit/push後の実SHAとremote一致は新runの`PUBLICATION_RECEIPT.json`に保存する。

**再現。** 既存Python `.venv-server-v2-20261013/bin/python`を使い、`PYTHONDONTWRITEBYTECODE=1`、`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`を設定する。新しいexclusive診断rootで`preservation.capture_snapshot`を先に実行し、次に下記を実行する。元runや報告の上書きは拒否される。詳細は [REPRODUCE.md](REPRODUCE.md)。

```text
python -B -m offline_evaluation.analysis.goal015b.audit \
  --run-root <new-exclusive-goal015b-run> \
  --report-root <new-exclusive-goal015b-report-directory>
```
