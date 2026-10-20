# Goal015C 保存済み12件の実測結果

Goal015Dでは、利用者指定のplan・ledger・evaluation・summary・GitHub verificationのSHA256を照合し、予定12行と保存結果の対応を確認した。生成、評価、GitHub再取得・復元は繰り返していない。原本の正式判定をそのまま集計した。

| Builder | A: 現在の順序 | B: 元のproperties順 | C: JSONをtextで指示 |
|---|---:|---:|---:|
| OpenAI | 0/2 | 2/2 | 2/2 |
| Anthropic | 0/2 | 2/2 | 0/2 |

表は`task_pass / 予定数`である。全12件は応答完了・usage既知。protocol-validは10/12、8セルの課題成功は6/12、指定された3操作列への適合も6/12だった。これらは別々の判定であり、同義ではない。Aの4件はprotocol-validのcarve-onlyで、最終voxel数は0だった。Bの4件とOpenAI Cの2件は最終8voxel・課題成功・3操作列適合だった。

Anthropic Cの`goal015c_call_002`と`011`は原本どおり`json_failure`。各保存textのhashを確認してから本文を読み、両方で先頭・末尾のMarkdown code fenceを確認した。fenceを含む本文全体は厳密なJSON文書ではない。fenceを除去した内部のJSONを解析・修復・再評価しておらず、仮想的な成功数は追加していない。その他10件の保存textは再読していない。

実ledgerをDecimalで合計したGoal015Cの既知実費は**USD0.01865750**で、12件すべてが対象である。親Goal015の既知費用USD4.43765700、Goal015Cの旧承認上限USD3.25、Goal015Dの新しい**未承認**提案上限USD5.00は別scopeである。上限を実費として扱わず、旧費用や旧枠を新実験の予算へ流用しない。

この合成課題・固定設定・2反復では、schemaのproperties順への介入に伴い、AからBへ成否が切り替わる観測が得られた。独立課題は**1種類**であり、12建築や外部独立追試ではない。Bはop位置に加えて座標keys等の順も復元するため、op先頭だけの効果を分離していない。旧72件の原因や反実仮想、provider内部のbug、一般的な100%成功、モデル間優劣は確定していない。

次の実建築2件は、親scheduleから重複を除いた最初の`buildings_100_v4_building_082`、`buildings_100_v4_building_003`を使うdevelopment canaryであり、未見holdoutではない。今後の非空生成とGTに対する再建築品質を分けて評価する。今回の集計を理由に旧72件を変更したり、旧残8件をBへ変更して補完したりしない。

既存の保存証明は`GITHUB_VERIFIED`、未公開Draft **418290999**、asset **659057302**、archive SHA256 `05b5a005ed49dc982541f2f48e6a97e986e487ed77d7fd99fae0ea090a51e908`。指定verificationのhash、復元済みplan IDとevaluation同梱を確認した。これは既存の再取得・安全復元receiptの検証であり、今回新たにネットワーク取得した証明ではない。Python環境はarchiveに含まれない。

全12行は[MINI_RESULTS_12.csv](MINI_RESULTS_12.csv)と[JSON](MINI_RESULTS_12.json)、hash・費用・群別集計・2失敗の読取記録は[MINI_RESULTS_SUMMARY.json](MINI_RESULTS_SUMMARY.json)に保存した。本報告処理の研究API送信・実承認・生成再実行は0。
