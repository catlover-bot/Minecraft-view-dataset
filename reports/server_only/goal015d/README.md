Goal015D：ミニ課題の結果確定と実建築2件・A/B16呼出しの未承認準備

保存済みGoal015Cの12件を確定し、実建築2件の独立したcanaryを未承認で準備した。今回の研究API送信、実承認、本番launcher実行、新規Draft操作はすべて0。旧Goal015の72件と残8件は変更していない。

| Builder | A task_pass / 予定 | B task_pass / 予定 | C task_pass / 予定 |
| --- | ---: | ---: | ---: |
| OpenAI | 0/2 | 2/2 | 2/2 |
| Anthropic | 0/2 | 2/2 | 0/2 |

12件はすべてSETTLEDでusage既知。protocol-validは10件、task_passは6件、3操作列適合は6件で、これらを同一の判定とは扱わない。Aの4件はprotocol-validなcarve-onlyで最終voxel数0。Bの4件とOpenAI Cの2件は小課題に成功した。Anthropic Cの002・011は保存textのhashを確認して読み、全体を囲むMarkdown JSON fenceを確認した。元のstrict JSON失敗を保持し、fence除去・修復・再採点は行っていない。

各callの返却model、finish、usage、費用、判定、操作数、voxel数は [12行CSV](MINI_RESULTS_12.csv)、[12行JSON](MINI_RESULTS_12.json)、[正式集計と保存証明](MINI_RESULTS_SUMMARY.json) に記録した。既存Draft418290999／asset659057302の保存証明は指定hash、plan ID、archive hashを確認して再利用し、再download・再復元していない。

この合成課題1種類・設定・各2反復では、properties順の介入に伴う成否の切替を観測した。Bはopだけでなく座標keys等の順も戻すため、op先頭だけの効果は未分離。旧72件の原因、provider内部bug、一般的な成功率、モデル間優劣は確定しない。[解釈](MINI_INTERPRETATION_JA.md) を参照。

次の対象は親scheduleの先頭から重複を除いたbuilding_082、building_003。既利用建築のdevelopment canaryであり、未見holdoutではない。FULL_RECORDの原bytesを再利用し、親両Builderのinput stringとも一致する。summary＋7フィールドを保ち、GT寸法・palette・管理metadataを加えていない。

| 建築 | 入力bytes | SHA256 |
| --- | ---: | --- |
| 082 | 2568 | a12a7a9cc16a4af90cecfff3fd88ebe96c6a5b567112623a7c2470a29ac610e6 |
| 003 | 2851 | a37fd73c5ac995cf0be5f834d258af076a65b87917dbb0c24349271a68f08a96 |

A/Bの差はset/fill/carveの3 branchのproperties列挙順だけ。型・値、required/anyOf、prompt、同一providerのinput string、モデル設定・capは共通。Bを再ソートした送信bodyは拒否する。[入力照合](INPUT_BINDING_PROOF.json)、[固定仕様・全16行の順序](EXPERIMENT_SPEC.md)、[8 wire](wires/) を参照。

新plan IDは `ed55298e3e544cb24a39338d098946afe313cd2238ee815016f0031c3f7d59b5`。canonical planは `/root/minecraft-runs/goal015d-20261020T114450Z-ebe767ac/plan.json`、SHA256は `a1b9db528c68fd9a0cebe5238ba65aad4cad39de7fb83cb578f72e3285ae739e`。[同一bytesのplan公開コピー](CANARY_PLAN.json) と [実行binding](EXECUTION_BINDING.json) に、76実装依存・仕様・入力・wire・順序・親来歴を束縛した。

| 費用区分 | USD | 状態 |
| --- | ---: | --- |
| 親Goal015 | 4.43765700 | 72件の既知費用。残8件は保留 |
| Goal015C | 0.01865750 | 12件のledgerをDecimal集計した実費 |
| 新Goal015D | 5.00 | 独立16件の提案上限。未承認 |
| 新Goal015D全16件の保守予約 | 4.235008 | OpenAI 8×0.165536＋Anthropic 8×0.363840 |

予約は実費予測ではない。入力全context＋output capの保守値で、400000入力tokensを実送信可能量と主張せず、最大入力と最大出力の同時達成も仮定しない。旧3.25／未承認7.00枠や旧1.10外挿を流用しない。各隣接A/Bペアの両行を最初のPOST前に耐久予約し、known usageだけ精算する。UNKNOWN時は相方のNOT_SENT予約も保持して停止する。

Goal015Cの同契約の公式取得本文・review・sealを検証して期限内再利用した。原取得日時を変更していない。現証拠の期限は **2026-10-21 10:00:58.172322 UTC**（Vilnius 13:00:58.172322）。今回の新規公式GETは0。[料金証拠の来歴](PRICING_REUSE.json) を参照。

利用者が後でtmux内の実TTYから起動する1行は次のとおり。この作業では実行していない。

```bash
bash /root/minecraft-runs/goal015d-20261020T114450Z-ebe767ac/start-reviewed.sh
```

launcherはcanonical path、未開始、commit済みcode/registry、plan/input/wire hash、料金期限、親・旧補完・Goal015Cとの競合をキー入力前に検査する。不足キーだけ非表示入力し、再度preflight後、plan IDとUSD5.00を単一関数で作る承認文を表示する。利用者の実TTY承認がなければledger/authorityを作成しない。承認をpipe・自動入力しない。tmux固有の起動義務を追加せず、実TTYを要求する。

期限後はそのまま起動できない。同契約の新規公式GET→本文review→sealを新しいdirectoryに保存し、`--pricing-evidence /absolute/path/to/NEW_REVIEWED_EVIDENCE.json` を同じlauncherへ渡す。取得・seal用の既存CLIは `offline_evaluation.experiments.goal015c.pricing`、Dの`pricing` moduleは実行CLIではない。日時の書換えやplan複製で期限・予算をresetしない。モデル・価格・契約の変更時は同planとして進めない。

実行器は最大16件、concurrency1、retry0。HTTP・認証・契約不一致・usage不明は保全して停止し、同callを自動再送しない。known output failure、正常empty、carve-onlyは結果として保存して続ける。旧残8件や本実験への自動移行はない。

将来の評価はraw→独立text抽出→strict validator→凍結executorと独立executorの照合を通す。16行・8セル・8ペアを保存し、emptyをprotocol-invalidへ変更せず、欠測gridやfailure0を作らない。実建築へ小課題のtask_passを流用しない。既定の6指標だけを元occupancy最適shiftで共有評価し、GTはこの評価段階でだけ読み込む。B−Aは共通に定義された値だけ。2建築からbootstrap・p値・一般的ランキングを作らない。

16件完了後は評価→新規未公開Draft保存→再取得SHA検証・安全復元まで進む。exit0や16 SETTLEDは建築品質の成功を意味しない。完了にはstatus JSONの`COMPLETE_SAVED`と保存証明を確認する。partialとUNKNOWNを隠さない。

保存だけの障害は同checkpointを使う`finish`／`save-only`で回復し、生成を再開しない。次は利用者が保存障害の後に、repository directoryから実行する回復例である。

```bash
/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python -B -m offline_evaluation.experiments.goal015d.cli finish --plan /root/minecraft-runs/goal015d-20261020T114450Z-ebe767ac/plan.json --allow-create
```

archiveは新plan、controls、入力、wire、raw、評価、実装、仕様、料金資料、必要な2建築のGTを束縛する。Python3.12と既存venv、深い旧来歴の原本等は外部依存であり、archive単独の環境を含む完全再現は主張しない。既存Draftを変更・公開・削除しない。

新差分に限定した最終suiteは **112 PASS、fail/error/skip 0**。CPU32.882694秒、elapsed36.481215秒、最大RSS80,708KiB。[JUnit](FINAL.junit.xml) と [テストreceipt](FINAL_TEST_RECEIPT.json) に実装hashとコマンドを残した。socket/DNSを遮断し、明示した非秘密環境と合成応答・mock GitHubを使用した。実PTY fixtureの承認はfixture専用で、本番の実承認ではない。旧suite・旧全rawの再解析、新規install、OS/SSH/Codex設定変更は行っていない。

追加の限定mockでは実際に束縛した76実装・2 FULL入力・8 wire・4 GT・11料金本文を用い、合成empty応答16件の評価、download失敗、同checkpoint再試行、save-onlyを確認した。173ファイルを安全復元し、132の束縛sourceは前後不変。mock Draft/assetは各1件、実ネットワークattemptは0だった。[保存監査](FINAL_CHECKPOINT_AUDIT.json)、[実行したmock scriptの同一コピー](BOUND_DEPENDENCY_MOCK.py)、[プロセス資源](BOUND_DEPENDENCY_MOCK_RESOURCES.txt) を参照。CPU8.57秒、elapsed8.72秒、最大RSS83,212KiB。この合成結果を実canaryの成績へ数えない。

開始UTCは2026-10-20T11:44:50.106659Z。終了UTC、限定した旧sourceの開始/終了hash、最終監査、commit後のread-only preflight、remoteと作業状態は [MANIFEST](MANIFEST.json)、[保全](PRESERVATION.json)、[要求別監査](REQUIREMENT_AUDIT.json)、[commit後preflight](POSTCOMMIT_PREFLIGHT.json) を参照。研究API送信0、実承認0、本番launcher実行0のまま準備を終了する。
