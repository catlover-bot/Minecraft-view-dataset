# Goal006B：認証失敗の記録と補完実行の準備

Goal006B の準備・監査では有料研究 API への送信は **0 回**。新しい OpenAI API キーへの更新は利用者からの申告として扱い、キーの値を表示・保存せず、有効性確認の API リクエストも行っていない。ここにある CPU/fixture の成功は新しいキーの有効性を証明しない。

元の実験は `unknown_usage:auth_failure` で停止した状態を保持する。補完は別の plan・approval・authority・ledger で行う設計で、元の停止理由や失敗を消して再開しない。今回の作業では補完送信を承認・開始していない。

## 実台帳から確認した状態

[accounting.json](accounting.json) と [auth_failure.json](auth_failure.json) の監査時刻は 2026-10-15 19:41:59 UTC。

| 区分 | 件数 | 意味 |
| --- | ---: | --- |
| 継続 800 件のうち試行済み | 781 | 成功 761 + usage 既知の失敗 19 + HTTP 401 の失敗 1 |
| usage 既知の失敗 | 19 | truncation 16、validation failure 3。今回の補完対象にはしない |
| HTTP 401・usage 不明 | 1 | `building_085/O` の OpenAI 生成。既に送信した失敗 |
| 未実行 | 19 | 元 schedule の global index 941–959。送信も予約 claim もまだない |
| 先行評価分を含む試行済み枠 | 941 / 960 | 先行 160 + 継続 781。成功数や完成済みデータセット数ではない |
| 先行履歴を含む累積送信 | 971 | 先行 190 + 継続 781。履歴の retry 4 を含む |

HTTP 401 の対象は `buildings_100_v4_building_085`、condition `O`、description model `openai`、生成 provider `openai`、model `gpt-5-mini`。元の local index は 780、global schedule index は 940、attempt ID は `2a912564cd574eeea2f6fa75d1683420`。2026-10-15 18:48:12.433510 UTC に `failed/auth_failure` として記録された。

HTTP 401 と保存済み payload hash・immutable claim の一致は確認済み。元の HTTP error transport は body と provider error code を保存していないため、失効した具体的な資格情報・project・organization や請求の有無は保存証跡だけでは確定できない。raw artifact・response ID・response model・usage はこの 1 件にはない。HTTP 401 を根拠に usage を 0 に書き換えない。

## 費用と予約の区別

| 数値 | USD | 扱い |
| --- | ---: | --- |
| usage 既知 970 送信の価格計算額 | 20.857851 | 凍結した標準料金での保守的な計算。実請求額ではない |
| HTTP 401 の usage 不明予約 | 0.165536 | 入力 400,000 + 出力 32,768 tokens の保持額。実測 usage・請求額ではない |
| 現在の予算消費として扱う合計 | 21.023387 | 上記の計算額と未解決予約を両方含む |
| 補完 20 件の追加最大予約 | 5.293760 | OpenAI 10 + Anthropic 10、入力・出力の最大予約合計 |
| 追加 20 件を全額予約した場合の累積 | 26.317147 | 元の USD 30 上限以内。古い不明予約を解放しない |

既知 usage は 9,132,037 tokens、HTTP 401 の未解決予約は 432,768 tokens、予算上の合計は 9,564,805 tokens。実請求額は `UNKNOWN_NO_INVOICE_OR_BILLING_RECONCILIATION`。補完 20 件の予約は 6,655,360 tokens で、実消費予測ではない。

20 件を送信すると累積送信は最大 **991** になり、元の上限 990 を 1 回超える。旧承認を流用せず、認証失敗の明示的な再試行 1 回を新しい承認範囲に含める。履歴 retry 4 は保持し、追加の手動認証再試行は最大 1、自動 retry は 0。単に USD 30 以下であることを追加送信の承認とは扱わない。

## 保存した原本と GitHub checkpoint

元の canonical plan は `/root/minecraft-runs/goal006-remaining800-20261014T101309Z-v2/plan.json`。identity は `db5533bffb583d0b1efcbcdb7c1563edf2ac87606334e6ce6cd766c650075843`。

| 原本 | SHA-256 |
| --- | --- |
| plan | `db78ec8e59b5921e248af42a7c8390647bd549c09c11e055c144729a247a2dce` |
| ledger | `6c301ecb572f232bc8464bcbbd42ff229ac588929bb8497601fefc063298ff35` |
| approval | `fdf5878be4122a686313ce926d00ce60a04973038126812f3998f814e446c07a` |
| authority | `5cd9ffa0c13491be47eb9b0ed36a1a3acc9af41efd730a983a9b6bbe13e0257e` |

[checkpoint_audit.json](checkpoint_audit.json) は 320・480・640・781 件時点の既存ローカル package と既存 GitHub 検証記録を照合した読取り監査。[existing781_github_verification.json](existing781_github_verification.json) は、その後に **781 件の最新 Draft を GitHub から新しく取得して検証した証跡**。両者の確認範囲を区別する。

最新 Draft Release ID は **414157245**、tag は `checkpoint-goal006-db5533bffb58-0781-20261015T184855214460Z-e17ef2a5`、target commit は `0b0175b1fe3a75da637a1d65716986e7b909fd5b`。ローカル package は `/root/minecraft-checkpoints/goal006-db5533bffb58-0781-20261015T184855214460Z-e17ef2a5`。

| Asset | Numeric ID | bytes |
| --- | ---: | ---: |
| `continuation-point-in-time.tar.gz` | 644830729 | 34,591,289 |
| `FILES.json` | 644830909 | 643,283 |
| `snapshot.json` | 644830956 | 613,581 |
| `SOURCE_REFERENCE.json` | 644830982 | 1,029 |
| `RESTORE.md` | 644831043 | 445 |
| `SHA256SUMS` | 644831088 | 422 |

archive SHA-256 は `c7897e5f121df0c08cc2a2a7e993f02fae3ac15fa5ca6f248e1e16cd14ca36a5`。2026-10-15 19:43:06 UTC の再検証では全 6 assets の name・size・SHA・GitHub digest を照合し、新しいディレクトリへ安全に展開した。3,093 files の復元、781 claims、2,302 raw/parsed/voxel artifacts を検証し、snapshot が参照する原本 3,092 files の hash は処理前後で一致した。copied plan の canonical path 拒否も確認した。

取得先は `/root/minecraft-checkpoints/goal006b-operations/existing781-20261015T194216271621Z-c9957826`。数値 Release/asset ID による GET のみで、GitHub mutation と再アップロードは 0。既存 Draft は保持している。

## 別台帳による補完

実装は [planning.py](../../../offline_evaluation/operations/goal006b/planning.py)、[runner.py](../../../offline_evaluation/operations/goal006b/runner.py)、[checkpoint.py](../../../offline_evaluation/operations/goal006b/checkpoint.py)、[cli.py](../../../offline_evaluation/operations/goal006b/cli.py)。元の実験コードの凍結対象外に追加し、実際の payload builder・parser・validator・executor は既存実装を利用する。

補完 schedule は元の global index **940–959** をそのまま保持する。最初の 1 件に `manual_auth_retry` と元 attempt ID を記録し、残り 19 件は `first_attempt` と記録する。model ID、accepted response model IDs、prompt、schema、入力、条件、GT、timeout、output 上限 32,768、順序と provider ごとの payload を元 plan と比較し、変更された plan を拒否する。concurrency は 1。

新しい `goal006b-.../plan.json` は `UNAPPROVED_SUPPLEMENT20`、`api_allowed: false` で作成する。別 lineage の authority により重複した有効化を拒否する。実行中は元の lock と補完側 lock を保持し、元の全 snapshot files・inventory・controls を照合する。最初の予約前には元の 781 件 GitHub checkpoint を改めて検証する。原本・scope・payload・claim・台帳に不整合があれば送信しない。

新たな 401・timeout など usage 不明の場合は、その予約を保持して補完台帳を停止する。予約済みのまま中断した試行も自動再送しない。usage 既知の truncation/validation failure は失敗として 1 回記録して次の固定入力に進み、再試行を追加しない。保存失敗後の再実行でも、既存の送信・claim を消してやり直さない。

補完結果は元の失敗と別に保存する。将来、全補完が完了した場合でも、元の 401・既知失敗・実行時刻・補完由来を保持して条件付きの補完分析として報告する。元の 960 枠を無条件に完成済み成功データへ置換しない。今回、補完結果の生成や新しい全 960 枠の評価は行っていない。

補完の終了・停止時には新しいローカル checkpoint を作り、安全展開と SHA を検証する。その状態は **`LOCAL_VERIFIED` / `github_status: NOT_UPLOADED`**。これは既存 781 件の `GITHUB_VERIFIED` と別の状態であり、新しい補完結果が GitHub に保存済みであるとは報告しない。元の 781 件の response bytes は複製せず、`SOURCE_REFERENCE.json` から保持済み package を参照する。

## CPU / fixture 検証

最終結果は **51 unique cases PASS、最終 failure/error/skip は各 0**。初回の新規時刻失効テスト1件はfixtureが二つの時計の片方だけを進めたため失敗した。fixture修正後に当該1件だけ再実行しPASS。失敗した実行のXMLも残している。[集約記録](validation_summary.json)、[修正・再実行記録](runner_cli_finish.json)。

| 範囲 | 件数 | 証跡 |
| --- | ---: | --- |
| planning | 14 | [planning_tests.junit.xml](planning_tests.junit.xml) |
| runner | 21 | [runner_tests_initial.junit.xml](runner_tests_initial.junit.xml)、[runner_tests_boundaries.junit.xml](runner_tests_boundaries.junit.xml)、[runner_tests.json](runner_tests.json) |
| checkpoint | 10 | [checkpoint_tests.junit.xml](checkpoint_tests.junit.xml) |
| CLI | 6 | [runner_cli_finish.junit.xml](runner_cli_finish.junit.xml)、[runner_pricing_expiry.junit.xml](runner_pricing_expiry.junit.xml) |

主な確認対象は、旧承認・不正 scope・未承認での送信 0、固定 20 payload の一致、手動再試行 1 と自動 retry 0、元の unknown hold を含む USD 30 / calls 991 制限、GitHub 元 checkpoint 検証失敗時の送信停止、401/timeout/予約中断後の再送拒否、保存失敗後の再送拒否、copied plan・台帳改変の拒否、別料金証跡追記時の元 controls 不変、実 archive の生成・安全復元・secret scan である。

fixtures は合成した 781 件履歴と provider response を使い、live socket と provider HTTP を遮断する。履歴 source validator の境界を synthetic source 用に置き換え、新しい planning・approval・runner・台帳・payload・parser・会計・artifact 検証を実行する。実原本については上記の独立した監査と GitHub 再取得で裏付ける。200 GT、旧 1,200 outputs、先行 160 件の metrics 再計算は行っていない。

## 補完を実行する場合の手順と追加承認範囲

**以下の plan path と保存 commit は最終準備後に確定する。PENDING を残したコマンドは実行しない。**

- canonical 補完 plan: `PENDING_CANONICAL_GOAL006B_PLAN_PATH`
- 実装保存 commit: `PENDING_IMPLEMENTATION_COMMIT`
- 実際の disabled plan 作成・preview・原本不変の最終証跡: `PENDING_FINAL_CPU_AUDIT`
- branch push と remote SHA 照合: `PENDING_GITHUB_SAVE_VERIFICATION`

通常の確認は `--preview` を使用する。mode 省略時も preview で、approval・authority・ledger の作成や生成送信を行わない。未有効化 plan に対する `--checkpoint` は拒否し、その操作によって有効化用 directory を先に作らない。

```bash
/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python -B /root/workspace/Minecraft-view-dataset/offline_evaluation/operations/goal006b/cli.py --plan PENDING_CANONICAL_GOAL006B_PLAN_PATH --preview
```

実行日の公式料金・model・出力上限の証跡が古い場合は、同じ条件の当日確認 JSON を **補完側だけ** に追記する。元 plan の pricing revision は変更しない。取得した公式証跡と frozen terms が異なる場合はこの補完 plan で実行せず、条件を再検討する。

```bash
/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python -B /root/workspace/Minecraft-view-dataset/offline_evaluation/operations/goal006b/cli.py --plan PENDING_CANONICAL_GOAL006B_PLAN_PATH --refresh-pricing PATH_TO_CURRENT_UTC_OFFICIAL_PRICING_JSON
```

将来、有料生成を明示的に承認する場合の実行コマンドは次の 1 本。今回の作業では実行しない。

```bash
/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python -B /root/workspace/Minecraft-view-dataset/offline_evaluation/operations/goal006b/cli.py --plan PENDING_CANONICAL_GOAL006B_PLAN_PATH --execute
```

`--execute` は両 provider キーの存在だけを確認し、API probe は行わない。新規 activation は `/dev/tty` で plan identity を含む確認文の完全一致を要求する。必要な追加承認は、**未実行 19 件 + 元 HTTP 401 の明示再試行 1 件、最大 20 新規送信、累積 calls 991、履歴 retry 4 + 手動 1、自動 retry 0、元の未解決予約を保持した累積 USD 30、出力 32,768、固定条件**。scope は `goal006b_19_unattempted_plus_1_explicit_auth_retry_cumulative_usd30_calls991`。

既存キーが更新されたという申告だけではこの追加有料範囲は承認されない。サーバー上の別台帳・checkpoint 用ディレクトリへの書込みと GitHub 読取りに sandbox の対象別承認が必要な場合は、それぞれの具体的操作について審査する。補完を開始するために既存原本を編集・reset・再 clone する手順はない。
