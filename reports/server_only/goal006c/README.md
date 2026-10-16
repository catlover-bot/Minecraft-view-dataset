# Goal006C：HTTP520停止からの保全・残17件の準備結果

Goal006B1 の補完は 3 件目の HTTP 520 / usage 不明で停止した。Goal006C は元の 781 件、補完 3 件、HTTP 401 と HTTP 520 の停止記録・予約を保持し、補完 3 件の GitHub Draft 保存と、本当に未試行の global index **943–959 の 17 件だけ**を扱う別 plan の準備を完了した。**この Goal の有料研究 API 送信、token-count API、認証・課金 probe、HTTP 520 の再送はすべて 0 回**。実 TTY での有料承認入力も代行していない。

## 実原本の読取り監査

[initial_integrity.json](initial_integrity.json) は実 canonical plan と台帳に既存 validator を適用した証跡。元と補完の run / authority lock を読取りハンドルで取得し、scope・identity・料金条件・payload・claims・成功応答の parse/voxel 整合性を検査した。元 781 件の応答検証は 1 回、補完の成功応答検証は 2 件。旧 1,200 outputs、200 GT、先行 160 件の metrics を再計算していない。

作業開始 HEAD は `24f72a06319415e34acbd5efeabf7870f35e4d2d`、専用 branch は `research/server-only-20261013`。main は `7309327efd1c19dda793b19e8d8e6ce0b5868db9`。root、ホスト `88.216.191.99`、Ubuntu 24.04.5 LTS、`/root/workspace/Minecraft-view-dataset` を確認した。開始時の生成プロセスは 0、協調 lock はすべて取得できた。実効設定は `workspace-write` / `auto_review`。OS・SSH・ユーザー・権限設定を変更せず、設定値だけから OS レベルの隔離を保証したとは扱わない。

| 対象 | 実台帳の状態 |
| --- | --- |
| 元 Goal006 継続 | 781 / 800 試行。成功 761、usage 既知の失敗 19、HTTP 401 の usage 不明 1 |
| 元の停止 | `unknown_usage:auth_failure`。global 940 の元 attempt を保持 |
| Goal006B1 補完 | 3 / 20 試行。成功 2、HTTP 520 の usage 不明 1。未試行 17 |
| 補完の停止 | `unknown_usage:http_failure`。global 942 を再送しない |
| 通算送信 | 974 = 元の累積 971 + 補完 3 |
| retry | 歴史的 4 + 明示的な認証再試行 1。自動 retry 0 |

元の 19 未実行枠のうち global 941 と 942 は今回までに試行済みになったため、Goal006C の未実行枠は 17。global 940 の元 401 と補完成功は同じ入力枠に対する別の試行として残す。

| B1 local / global | provider・model | scene / condition / intent | 結果 | usage | 保守的価格計算 / 予約 USD |
| --- | --- | --- | --- | --- | ---: |
| 0 / 940 | OpenAI `gpt-5-mini` | `building_085` / O / `manual_auth_retry` | success | input 1,792、output 7,255 | 既知 0.014958 |
| 1 / 941 | Anthropic `claude-haiku-4-5-20251001` | `building_085` / O / `first_attempt` | success | input 2,569、output 1,334 | 既知 0.009239 |
| 2 / 942 | OpenAI `gpt-5-mini` | `building_085` / J / `first_attempt` | failed、HTTP 520、`http_failure` | 不明 | 予約 0.165536 |

scene の完全名は `buildings_100_v4_building_085`。成功 2 件はそれぞれ raw / parsed / voxels の全 3 artifacts と immutable claim があり、hash・size・応答内容と保存された派生物の整合性を確認した。HTTP 520 の attempt は `2767df87e4a5466f953fe356ff7a708c`、claim はあるが raw / parsed / voxels はなく、usage・response model・response ID もない。HTTP body やキーを出力していない。

元と B1 の台帳内に重複 provider/input pair や plan 外入力はない。両台帳間の重複は明示承認された global 940 の認証再試行だけで、未試行 17 件はどちらの試行済み pair とも重ならない。

## 二つの usage 不明予約と予算

| 区分 | USD | 解釈 |
| --- | ---: | --- |
| 元の usage 既知分 | 20.857851 | frozen rate による保守的な価格計算 |
| 補完成功 2 件 | 0.024197 | 実 usage を使った保守的な価格計算 |
| 既知分の合計 | 20.882048 | 請求書の確認額ではない |
| 元 HTTP 401 の不明予約 | 0.165536 | 元台帳に保持 |
| 補完 HTTP 520 の不明予約 | 0.165536 | B1 台帳に保持 |
| 不明予約の合計 | 0.331072 | 実測 token / 請求額ではない |
| 現時点の累積予算計上 | 21.213120 | 既知価格計算 + 不明予約 2 件 |
| 未試行 17 件の最大追加予約 | 4.598848 | 各 fixed payload の入力・出力上限から再計算 |
| 17 件を全額予約した場合の累積 | 25.811968 | 元の累積 USD 30 以下 |

実請求額は `UNKNOWN_NO_INVOICE_RECONCILIATION`。520 の発生経路、モデル側での実行有無、料金が発生したかは保持証跡から確定できない。HTTP status を根拠に「無料」「未実行」「usage 0」と判断せず、二つの保留予約を解放しない。新しい 17 件に自動 retry・手動 retry を追加せず、累積送信上限は **974 + 17 = 991**。

## 原本不変と保存対象

読取り監査の前後で **3,865 files** の SHA と file inventory が一致した。対象は元の source snapshot 3,740 files、B1 の plan / approval / authority / ledger / claims / response artifacts、旧 disabled B plan、指定補完 checkpoint の restored tree を含む全 25 files、旧 Goal006B/B1 reports、凍結実装 44 files の和集合。元の 401 と B1 の 520 の halt を含め、削除・上書き・巻戻し・再初期化はない。実行時間は約 29.91 秒。

| 保持する plan | SHA-256 / identity |
| --- | --- |
| 元 `/root/minecraft-runs/goal006-remaining800-20261014T101309Z-v2/plan.json` | SHA `db78ec8e59b5921e248af42a7c8390647bd549c09c11e055c144729a247a2dce`、identity `db5533bffb583d0b1efcbcdb7c1563edf2ac87606334e6ce6cd766c650075843` |
| B1 `/root/minecraft-runs/goal006b-ttyfix20-20261015T201419244530Z/plan.json` | SHA `088e7f5253c078ea310a7130e88c23df602a26910a61a0785266fd8af3c4ed21`、identity `206dfda6468a90b8d67700e3799571ff0de5aabe1263c15754efe242887f8922` |

元 781 件の Draft Release **414157245** を保持する。その `GITHUB_VERIFIED` は、補完 3 件の GitHub 保存状態とは別である。

補完 3 件の既存ローカル checkpoint は `/root/minecraft-checkpoints/goal006b-206dfda6468a-0003-20261015T231926513964Z-0aaff2c5`。archive は `supplement-point-in-time.tar.gz`、SHA-256 は **`85498f3b9492d4fc37d9778c1e2f7261bf7bddf0ca6d6e20bc5aeb2410592afe`**。開始時点は `LOCAL_VERIFIED` / `NOT_UPLOADED`。この監査では全 6 assets の local hash / size と snapshot の原本参照を検証し、過去のローカル安全復元記録（17 files）を確認した。今回の新しい復元・GitHub 検証と混同しない。

| Goal006C の保存証跡 | 状態 |
| --- | --- |
| 新しい Draft Release ID | **414396750**、`draft: true` |
| 固定 tag | `checkpoint-goal006b-206dfda6468a-0003-20261015T231926513964Z-0aaff2c5` |
| target commit | B1 実装 `28fa9ea2f9bae114cea2be71f352ea6c955a5eaa` |
| asset ID で新規取得した検証 directory | `/root/minecraft-checkpoints/goal006c-operations/b3-save-20261016T001742Z-4d1a08e6` |
| manifest・ledger・claims・実 artifacts の復元照合 | **VERIFIED**、manifest 掲載 17 files + `FILES.json` 自体（実ファイル計 18）/ 3 claims / 6 artifacts |
| 元 ledger と復元 ledger の SHA | `f3ae55def9d780dc3a9614406aa5d4a0382d52f3112f6edd71584107f65c5864` |

| Asset | Numeric ID | bytes |
| --- | ---: | ---: |
| `supplement-point-in-time.tar.gz` | 645596888 | 364,769 |
| `FILES.json` | 645596910 | 5,221 |
| `snapshot.json` | 645596937 | 7,070 |
| `SOURCE_REFERENCE.json` | 645596954 | 1,790 |
| `RESTORE.md` | 645596969 | 464 |
| `SHA256SUMS` | 645596989 | 420 |

2026-10-16 00:18:07 UTC に **`GITHUB_VERIFIED`** を確認した。[b3_github_verification.json](b3_github_verification.json) に全 6 assets の SHA / size / GitHub digest と復元結果、[b3_save_operation.json](b3_save_operation.json) に保存専用実行を記録した。新しい Draft へ保存した全 6 assets を numeric asset ID で再取得し、全 bytes の照合・secret scan・manifest・source references・実際の claims と artifacts の復元整合性を検証した。copied plan の起動拒否も確認した。公開・削除・同名 asset 上書きはない。

[保存した Draft Release](https://github.com/catlover-bot/Minecraft-view-dataset/releases/tag/untagged-9579207d15b1f62dcfdd) は、00:20:25 UTC の [最終読取り照合](b3_release_final_read.json) でも draft / unpublished を維持し、全 6 assets の identity は不変だった。

元の `checkpoint.json` は作成時の `LOCAL_VERIFIED` / `NOT_UPLOADED` の bytes をそのまま保持する。今回の GitHub 保存完了は別の上記検証証跡に記録し、原本 metadata を更新していない。

新規運用コードは `offline_evaluation/operations/goal006c/` に分離する。元 Goal006 / B1 が束縛する Python ファイルを変更・追加せず、既存 hash 検査を保持する。新しい保存ヘルパーは B checkpoint schema を検査し、tag の 404 を根拠に新規作成しない。既存 Draft は数値 ID または全ページ一覧から照合し、一致すれば再利用、不一致は停止。全 6 assets の secret scan・digest 照合・新規取得・安全復元が揃うまで `GITHUB_VERIFIED` としない。raw model responses は tracked files に含めず、Draft assets にのみ保存する。

[独立 planning / CLI レビュー](code_review.json) では、固定 scope、全 payload bytes 比較、元 source と checkpoint 証跡の hash 束縛、料金条件、実 TTY 承認、保存専用の制御フロー、実装 commit と全 code bytes の一致検査を確認した。旧凍結コード 44 と Goal006B/B1 reports 38、合計 82 files の hash も一致した。これは静的レビューであり、実 GitHub 保存とテスト結果はそれぞれの証跡で確認する。

## 新しい未承認 17 件 plan

新 plan は **global 943–959 の 17 件だけ**を元と同じ payload、model、prompt、schema、output 上限 32,768、timeout、concurrency 1、評価器、GT で扱う。global 940・941・942、既知の truncation / validation failure の再送は scope 外。元 781 件と B1 3 件を source references と hashes で保持する。

別の plan / approval / authority / ledger に結び付け、元の approval は流用しない。新しい exact `/dev/tty` 承認、当日 UTC の公式料金条件確認、両 provider キーの存在、元 781 件と補完 3 件の両 checkpoint の復元検証を最初の送信前に必要とする。Goal006C 中には有効化しない。

| 新 plan と実装 | 状態 |
| --- | --- |
| 先に push して固定した実装 commit | `0ce29a81b27362862977b2f85c303a26883781cf`、remote 一致 |
| canonical plan 絶対パス | `/root/minecraft-runs/goal006c-remaining17-20261016T001844Z/plan.json` |
| plan SHA-256 | `79e24a6a494bc69fedc2b7c7d4319e5daad946dda4d56fbd0cf77131c1866c69` |
| plan identity | `03340ae00dab4ab13be7cc8e9de379652a0ea92cf2cec2edda38e006c2668e4c` |
| 別承認 scope | `goal006c_unattempted17_global943_to959_cumulative_usd30_calls991_keep_both_unknown_holds` |
| disabled / approval なし / ledger なし | [new_plan.json](new_plan.json)、[preview.json](preview.json) |

実装 push と補完 3 件の独立した GitHub バックアップ検証を先に完了し、2026-10-16 00:18:50 UTC に新 canonical plan を作成した。実データの 17 payload bytes を元 plan と比較して全件一致を確認し、preview は `NOT_RUN` / `approved: false` / `api_allowed: false`。canonical directory にあるファイルは `plan.json` だけで、enabled directory と authority はない。開始会計は 974 calls・USD 21.213120、usage 不明 2 件を保持する。準備と preview 等の検証は約 32.65 秒で、追加 API と承認入力は 0。

監査時点で元 Goal006 と B1 の料金証跡は 2026-10-15 のものであり、UTC 2026-10-16 の freshness 検査は期待どおり失敗した。元の価格条件の妥当性検査は通るが、実行日の新しい公式確認は別に必要。

2026-10-16 00:03:53–00:07:49 UTC に公式 8 pages の該当箇所を取得・確認し、[当日の公式料金証跡](pricing_20261016/official_pricing.json) を作成した。SHA-256 は `6c4a4288bca7a12c751487a9e4c8c1f91710b4b6cad54a29f80ae9960fde2a2b`。凍結した model・料金・context / output 条件に変更はない。元の料金記録や hash は変更せず、同じ条件の取得証跡を Goal006C 側だけに使用する。後日 UTC 日付をまたぐ場合も Goal006C の pricing revision にだけ追記する。公式文書の確認はアカウントの利用可否やキーの有効性を API で検査したことを意味しない。

新しい unknown usage が発生した場合は、その試行と予約を保持して停止する。送信した試行をクリアして再開しない。保存失敗も台帳と成果物を残して停止し、保存のみの再試行手順を用いる。

## CPU / fixture と GitHub 保存

最終の統合 CPU/fixture テストは **95 件 PASS、FAIL 0、ERROR 0、SKIP 0**。内訳は planning / CLI 37、runner 29、archive / checkpoint 29。今回の全体実行は 82.48 秒、最大 RSS 62,692 KiB、BLAS / OpenMP は各 1 thread。[final_cpu_tests.json](final_cpu_tests.json) に結果と検査対象の hash を記録した。途中の部分実行には archive fixture の 1 件の失敗があり、その記録を残したうえで修正し、最終 95 件の実行ですべて PASS を確認した。途中の再実行や過去 Goal006B/B1 の件数はこの 95 件に加算していない。

検証対象は exact 17 payload / global indices、USD 30 / 991 calls と不明予約 2 件、別 plan の承認、二重起動、401 / 520 / timeout の再送拒否、Draft duplicate / digest 不一致の拒否、安全復元、copied plan の起動拒否。provider response は fixture / mock transport だけで、production `--execute` の起動と実研究 API 送信は 0。

[code_push_receipt.json](code_push_receipt.json) に実装 commit `0ce29a81b27362862977b2f85c303a26883781cf` の push、remote branch 一致、main 不変を記録した。新 plan の hash 束縛対象は元の 44 + 新規 Goal006C の 6 = **50 code files**。コードの push を新 canonical plan 作成より先に行った。

[final_integrity.json](final_integrity.json) は 2026-10-16 00:19:57 UTC の最終読取り検証。初期監査で記録した全 3,865 files と 6 protected directory の exact file inventory が一致し、元の 44 code files、両台帳の halt、USD 0.165536 の不明予約 2 件を保持している。新 plan の全 50 code hashes は実装 commit と一致し、17 payload bytes、disabled 状態、当日料金条件を確認した。保存後の全 6 downloaded assets と復元済み manifest の 17 files を再 hash し、元の成功応答や metrics は再実行していない。

この README と証跡は実装と別の reports-only commit で保存する。plan が束縛する実装 commit は `0ce29a81b27362862977b2f85c303a26883781cf` のままである。報告用 commit の最終 SHA、remote HEAD 一致、main 不変、worktree clean の確認結果は、その保存後の完了報告に記載する。

## 後日実行するコマンド

確認用コマンド。preview は有効化・承認作成・API 送信を行わない。

```bash
/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python -B /root/workspace/Minecraft-view-dataset/offline_evaluation/operations/goal006c/cli.py --plan /root/minecraft-runs/goal006c-remaining17-20261016T001844Z/plan.json --preview
```

未試行 17 件だけの追加有料実行を、利用者が後日明示承認する場合のコマンド。本 Goal では実行しない。

```bash
/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python -B /root/workspace/Minecraft-view-dataset/offline_evaluation/operations/goal006c/cli.py --plan /root/minecraft-runs/goal006c-remaining17-20261016T001844Z/plan.json --execute
```

17 件を実行した後も **global 942 の HTTP 520 による観測欠測は残る**。520 の再試行には、将来別の明示承認と別計画が必要。論文へ統合する際は成功、usage 既知のモデル失敗、運用上の usage 不明、認証再試行で補完された slot を分けて計上し、元の失敗を成功で上書きしない。
