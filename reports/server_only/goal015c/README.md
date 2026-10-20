Goal015C：schemaキー順序の局所検証と12呼出し診断の未承認準備

局所CPU検証、比較wire、未承認12件plan、実行器・評価器・保存経路・利用者用launcherを準備した。研究API送信、実モデル試験、実承認入力、実Draft操作はすべて0。親Goal015の72件（非空carve-onlyによる正常empty48、truncation22、訪問上限拒否2）は変更しない。carve-onlyの原因は未確定であり、親残8件とGoal015Aは保留のままである。

専用runは `/root/minecraft-runs/goal015c-20261020T094646Z-fcb2ec03`。本番planは `api_allowed=false`、`budget_approved=false`、approval/authority/ledger未作成である。以下のmock成果を実API成功として数えない。

**確定したキー順序の変換。** [LOCAL_ORDER_AUDIT.json](LOCAL_ORDER_AUDIT.json) は元schema、純粋payload構築、親の保存payload読込、送信前canonical clone、最終HTTP bodyを分けて記録する。保存payloadは既にソート済みだったため、保存物の読込からcloneへの追加の順序変化はない。一方、元の構築順を実際の旧send経路へ渡すisolated fixtureでは、`json.loads(canonical_json(payload))` が次の順へ変換し、捕捉した `Request.data` でも一致した。

| branch | 元properties順 | canonical clone後 |
| --- | --- | --- |
| set | op,x,y,z,block | block,op,x,y,z |
| fill | op,x1,y1,z1,x2,y2,z2,block | block,op,x1,x2,y1,y2,z1,z2 |
| carve | op,x1,y1,z1,x2,y2,z2 | op,x1,x2,y1,y2,z1,z2 |

required配列とanyOf配列（set,fill,carve）の順は不変。型・値のtreeは同じでも、順序を保持した表現とbody SHAは異なる。新adapterのBを送信直前に再ソートすると検証が拒否する。

旧O比較はGoal015Bで選ばれたbuilding_003の両Builder各1件だけ。凍結した旧構築・serializerから再構成したbodyはop先頭を保ち、保存object_hashと一致する。ただし過去の完全wire bytesはなく、証明区分は **RECONSTRUCTED_FROM_FROZEN_CODE**。出力rawのキー順、APIの部分echo、ローカルRequest.data捕捉をremote受信の独立証明には使わない。親72応答の再診断は行っていない。

**固定課題と比較設計。** [EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md) に英語原文、正解JSON、判定表を実モデル試験前に固定した。全条件で同じsummary JSONと凍結builder promptを使う。fill→carve→setの3操作を旧executorと独立疎辞書executorで実行し、stone7＋glass1の8セルが元座標・材質で完全一致した。[TASK_CPU_PROOF.json](TASK_CPU_PROOF.json)

| 条件 | 許可した差分 |
| --- | --- |
| A：NATIVE_CURRENT_ORDER | Goal015と同じソート後properties順のnative schema |
| B：NATIVE_ORIGINAL_PROPERTY_ORDER | Aの3 branchのproperties列挙順だけ元へ戻す |
| C：TEXT_JSON_INSTRUCTION_ONLY | Aからtext.format／output_config.formatと空になった親objectだけ除く |

A/Bのrequired、anyOf、enum、型、説明、name、strictは同一。CにJSON mode、tools、回答例、JSON修復を追加しない。全条件の受信文字列を同じstrict validatorで評価する。primaryは最終grid完全一致で、3操作要求への適合は別列。空JSON・carve-onlyはprotocol-validでもtask_pass=false。material alias、移動、回転で正解へ修復しない。

予定順は `O/A, An/C, O/B, An/A, O/C, An/B`、次に `An/B, O/C, An/A, O/B, An/C, O/A`（O=OpenAI、An=Anthropic）。新ID `goal015c_call_001`〜`012`、concurrency1、retry0。2反復は予定された別試行であり、失敗の再試行ではない。出力失敗後も指定条件を続け、通信不明・認証/契約/モデル不一致・予算不足では停止する。

モデルはOpenAI `gpt-5-mini-2025-08-07`（reasoning medium、temperature/top_p/seed省略）、Anthropic `claude-haiku-4-5-20251001`（temperature1、thinking disabled）。output cap32768、timeout600秒、通常tier、画像・toolsなしを全条件で維持する。

**現時点の契約と独立した未承認予算。** 公開公式11文書の実本文、取得URL/UTC/SHA、読取り根拠と未確認事項をcanonicalへ保存した。[PRICING_REVIEW.json](PRICING_REVIEW.json) と [PRICING_EVIDENCE.json](PRICING_EVIDENCE.json) が実本文へ束縛される。captureと明示review/sealを分離し、日時だけの更新や自動reviewを認めない。

| Builder | 標準入力/出力単価（USD/百万token） | 1件の保守予約 | 件数 |
| --- | --- | ---: | ---: |
| OpenAI | 0.25 / 2 | USD0.165536 | 6 |
| Anthropic | 1 / 5 | USD0.363840 | 6 |

full-context入力予約＋cap出力による最大予約は **USD3.176256**、提案上限 **USD3.25、未承認**。実費予測ではない。OpenAI文書の最大入力272000に予約を縮めず、従来の400000入力予約を維持した。Anthropicは200000入力予約。cache割引は使わない。親既知費用 **USD4.43765700**、旧USD5、Goal015Aの未承認提案USD7とは別で、旧承認や残額の流用は0。

公開文書は現在の契約根拠であり、10/19当時のprovider内部を確定しない。Anthropicのnative format経路には追加promptと課金入力が入り得るため、Cとの差をconstraint decodingだけの効果とは呼ばない。アカウントの利用可能性、実際のremote挙動は未照会。固定OpenAI snapshotの文書上の停止日2026-12-11以降は拒否し、別モデルへfallbackしない。

**code・plan・wireの束縛。** [EXECUTION_BINDING.json](EXECUTION_BINDING.json) が67実装依存、仕様、6種類の最終wire、canonical planを固定する。6 bodyは [wires/](wires/) にも同一bytesを保存した。12 schedule行はこれらを参照する。

- plan ID：`95ef8e4c7e0d8e261087944374e1a690f32516caa91cf06fd95740a75b453503`
- plan SHA256：`065fe859b152912e41e0b5ef8697fa11c6e2764a862a7f23528654a8e7a21995`
- canonical plan：`/root/minecraft-runs/goal015c-20261020T094646Z-fcb2ec03/plan.json`

本番adapterは検証済みbytesをそのままRequest.dataへ設定する。正規化object hash、ordered表現hash、wire hashを区別し、送信直前のhash/bytes/Request.data一致をdurable記録してから公式endpointへPOSTする。headersとキーは記録しない。旧関数の契約をmonkey patchで解除していない。

**準備検証。** [FINAL_TESTS.junit.xml](FINAL_TESTS.junit.xml) は新規129件PASS、失敗/エラー/skip0。[FINAL_TEST_RECEIPT.json](FINAL_TEST_RECEIPT.json) は対象コードhashとCPU17.12秒、経過19.66秒、最大RSS54,984KiBを記録する。外部socket接続/DNSはfixture全体で拒否し、ダミー資格情報はisolated fixtureだけで使用した。旧全suiteは再実行していない。

順序保持、A/B受理集合、C差分、正解/empty/carve-only/truncation/touch拒否、通信不明、重複起動、scope/予算、EOF/Enter/誤承認文、保存only分離を検証した。launcherは構文とsynthetic実PTYで検証し、**canonical本番scriptは実行していない**。

[BOUND_DEPENDENCIES_FIXTURE.json](BOUND_DEPENDENCIES_FIXTURE.json) は実際に束縛した67実装と11文書を使った別のmock結合検証。12 mock応答→12行評価→archive→fake Draft/asset→再取得→安全な復元がPASSした。再取得失敗後の再実行でも同じcheckpoint・ledger bytes、fake Draft/asset各1件を維持した。実API・実承認・実Draft操作は0。

評価は独立抽出→strict validation→旧/独立executorを通す。全12行にHTTP/finish/usage、JSON、各操作数、3操作適合、voxel数、完全一致、truncation/touch拒否を保存し、known output failure・unknown・NOT_SENT・CPU評価不能を分ける。条件別の送信/完了/成功/費用分母を区別する。p値、bootstrap、ランキングを作らない。2反復から元72件の原因や一般的機構を確定しない。

**将来の利用者用起動。** 実在する専用launcherは次の1行。これは今回実行したコマンドではない。

```bash
bash /root/minecraft-runs/goal015c-20261020T094646Z-fcb2ec03/start-reviewed.sh
```

現在の料金証拠の期限は **2026-10-21T10:00:58.172322Z**（最初の実GET開始から24時間）。launcherは未開始状態、committed registry、実装/wire hash、親子同時実行、証拠期限をキー入力前に確認する。TTY read/writeは別ハンドル。承認文はplan IDと金額の単一生成関数から端末に表示されるので、その表示を利用者が確認して手入力する。Codexとscriptは承認を自動入力しない。

期限切れの場合は、新しい専用directoryへ `python -B -m offline_evaluation.experiments.goal015c.pricing capture --output NEW_DIRECTORY` を実行し、取得本文を実際に読んで新review JSONを作成する。その後 `pricing seal --capture CAPTURE_JSON --review REVIEW_JSON --output NEW_EVIDENCE_JSON`（同じmodule prefix）でsealする。既存Pythonは `/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python` を使う。

同契約の更新証拠はlauncherへ `--pricing-evidence /absolute/path/NEW_EVIDENCE_JSON` と明示して渡せる。plan複製・新予算resetは不要で、各送信前にも24時間を検査する。契約が変われば停止し、日時だけを変更して通さない。

将来の正常runは12件の状態確認後に評価と新規未公開Draft保存・再取得検証まで進む。途中停止はlocal checkpointを保全し、再開は確認済みauthorityの下でNOT_SENTだけ。unknownの予約は解放せず、同じIDを再送しない。exit0だけを完了証明に使わず、status JSONと保存verificationを確認する。COMPLETE_SAVEDは12 SETTLED・unknown0・NOT_SENT0と保存検証の完了であり、課題成功12件を意味しない。SAVED_ONLYも診断全体の完了とは別である。

将来、評価・保存だけを完了するCLIは、repository内から同じPythonで `-B -m offline_evaluation.experiments.goal015c.cli finish --plan /root/minecraft-runs/goal015c-20261020T094646Z-fcb2ec03/plan.json --allow-create`。既存checkpointの保存再試行だけなら `save-only --plan ...` を使い、新規診断Draft/assetの作成がまだ必要な場合は `--allow-create` を明示する。保存先を指定する場合は `--checkpoint-dir` に同じcanonical checkpoint、`--verification-dir` に新しいcanonical検証directoryを渡す。これらは生成を再実行しない。部分結果も12行で保存するが、PARTIAL_SAVEDは非zero終了。既存Draftの公開・削除・asset上書きはしない。

checkpointにはplan/authority/ledger、raw、12行評価、wire、prompt、仕様、実装、料金本文/reviewを含める。旧provenanceから参照されるさらに深い旧960件全体とPython環境は外部再現依存として明示し、今回のarchiveに全過去実験やvenvが含まれるとは主張しない。

**保全と保存。** [PRESERVATION.json](PRESERVATION.json) は開始snapshotの和集合76ファイルを終了時と照合し、親72 SETTLED/8 NOT_SENT/unknown0と費用、子disabled、診断未承認を確認する。新規コード・報告だけを専用branchへ通常commit/pushする。commit後の本番planの読み取り専用検証は [POSTCOMMIT_PREFLIGHT.json](POSTCOMMIT_PREFLIGHT.json)、公開copyは [PUBLIC_COPY_RECEIPT.json](PUBLIC_COPY_RECEIPT.json)、要件別証拠は [REQUIREMENT_AUDIT.md](REQUIREMENT_AUDIT.md)、最終hash集合は [MANIFEST.json](MANIFEST.json) を参照。

最終commitとremote一致・cleanの確認はcanonical `PUBLICATION_RECEIPT.json` と最終応答に記録する。実装・wire・planはその後も同一hashで検証される。旧8件、旧S/F80件、別実建築canaryへの自動移行はない。
