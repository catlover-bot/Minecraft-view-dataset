Goal006Aの保存復旧結果

既存320入力のDraft Releaseを再取得し、全6 assetsのSHA256・size・GitHub digest、安全展開、復元原本との一致を確認した。保存専用CLIの実行は終了コード0、`GITHUB_VERIFIED`。既存assetsの追加upload・上書き・削除・Draft公開は行っていない。今回の追加研究API送信は0回で、有料生成は再開していない。

**今回読み取ったサーバー状態**

2026-10-15 11:16:23 UTCの初期監査ではroot、作業場所`/root/workspace/Minecraft-view-dataset`を確認した。生成入口に一致する稼働プロセスはなく、既存の協調ロック10個を読取りで取得できた。原本・現在の承認・authority・台帳・予約・応答を検証した。[初期監査](initial_state.json)

| 項目 | 実測値 |
|---|---:|
| 継続入力の試行済み / 未試行 | 320 / 480 |
| 先行160件を含む本評価の試行済み / 全体 | 480 / 960 |
| 継続入力の成功 / 失敗 | 313 / 7 |
| 継続失敗の内訳 | truncation 7 |
| 現在の累積送信 / retry | 510 / 4 |
| 現在の累積tokens | 4,811,604 |
| 現在の累積保守的費用計算 | USD10.879646 |
| 未精算予約 / 不明usage | 0 / 0 |

継続分だけの使用量は320送信、retry0、2,983,123 tokens、USD6.830724。以前の190送信・USD4.048922は累積計算の親消費であり、現在値ではない。直近の応答完了は2026-10-14 15:06:49.641399 UTC。現在台帳の`halt_reason`はnullで、保存失敗は`checkpoint_save_failed / GITHUB_SAVE_FAILED`という運用上の停止として扱われている。

**既存Draftの保存検証**

[Draft Release 412856648](https://github.com/catlover-bot/Minecraft-view-dataset/releases/tag/untagged-ac267a2a8624283b2a9e)のtagは`checkpoint-goal006-db5533bffb58-0320-20261014T150741439937Z-ad2c6d7f`、target commitは`0b0175b1fe3a75da637a1d65716986e7b909fd5b`。ローカル`checkpoint.json`との一致、Draftの維持、専用ブランチとのcommit関係を確認した。

| asset | asset ID | bytes | ローカル・GitHub digest・再取得後の照合 |
|---|---:|---:|---|
| `continuation-point-in-time.tar.gz` | 640740758 | 14,838,426 | 一致 |
| `FILES.json` | 640740972 | 267,187 | 一致 |
| `snapshot.json` | 640741042 | 254,579 | 一致 |
| `SOURCE_REFERENCE.json` | 640741154 | 1,029 | 一致 |
| `RESTORE.md` | 640741284 | 445 | 一致 |
| `SHA256SUMS` | 640741416 | 422 | 一致 |

archive SHA256は`bb643bcd59872fbc4d869b0ba5628e585d54a4cc5827794aa8bfc42eba98b9cc`。復元した1,275ファイルのmanifest照合、320件の予約、946個の応答・解析・voxel成果物、固定順・payload・承認・authority・台帳の対応を検証した。復元台帳のSHA256は現在台帳と同じ`b437c5b999f3240562099a19f14a40c7586dd030d16321bbf20eb1d94f0edf69`。復元コピーからのplan読込みもcanonical制約により拒否された。

新しい再取得・復元先は`/root/minecraft-checkpoints/goal006a-operations/manual-20261015T112441847815Z-e26c8dca`。保存専用CLIは29.50秒で終了した。[全6 assets・復元の検証結果](existing320_github_verification.json)、[CLI実行・原本不変の記録](recovery_cli_execution.json)。

**保存失敗の診断**

現在の再現確認では、Release IDによる取得は終了コード0だが、同じDraftのtag取得は終了コード1・HTTP404だった。したがって「tagで見つからないのでDraftが存在しない」と扱う従来入口は、既存Draftの検索・再利用に不十分である。[GitHub読取りprobe](github_probe.json)

初回保存では6 assetsがGitHubに存在し、保存済みの再取得ファイルのうちarchive、`FILES.json`、`snapshot.json`の3個はローカルpackageと完全一致している。その後の正確な失敗処理・HTTP状態は残っておらず、初回失敗をtag検索、認証、通信障害などの特定原因に断定できない。

再試行時には再取得ファイルがなく、既存tag・target・Draft属性と一致する作成要求の準備ファイルが残っている。これは既存Draftを再利用できない経路に入った証拠となるが、当該試行の失敗HTTP状態や応答本文は保存されていない。今回確認したtag404と、過去の初回・再試行の失敗点を同一視しない。[過去2試行の保存済み証拠](historical_attempt_evidence.json)

**実装した運用入口**

追加コードは凍結hash対象外の`offline_evaluation/operations/goal006a/`に置いた。`archive.py`はRelease IDとasset ID、またはページネーション済み一覧でDraftとassetsを解決する。既存assetの内容が一致すれば再uploadせず、不一致・不足は具体的なstage、code、終了コード、HTTP状態を残して停止する。エラー記録には認証情報、ヘッダー、署名付きURL、GitHub CLIの生のstderrを含めない。

`resume.py`は初期320件のRelease ID・archive asset ID・SHAを検証したうえで、現在の台帳そのものと一致するcheckpointを検証する。後の480件や640件の保存が失敗した際に、古い320件の保存だけを根拠に進むことはない。送信済み・失敗済み・不明usageの入力は再送しない。

区間保存には凍結runnerの`checkpoint_callback`を利用する。既存ロックの保持中に増分評価、ローカルcheckpoint、GitHub保存・再取得検証を順に行い、検証が失敗するとその区間境界で停止する。失敗結果に出る`recovery_command`は保存専用であり、生成関数を呼ばない。

この入口の協調ロックは、この運用ヘルパー同士を直列化する。さらにrun直前にcanonical control filesのhashを再確認し、凍結runner自身の非ブロッキングロックを利用する。旧入口を別プロセスで同時実行した場合の、最終hash確認からrunnerのロック取得までの窓まで原子的に保護する構成ではない。再開には下記入口を使用する。

利用者が既存のAPIキーを設定した端末で実行する再開コマンドは次の1本。元の実TTY承認を再利用し、新規承認・plan再作成・台帳初期化は行わない。今回この有料再開コマンドは実行していない。

```bash
/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python -B /root/workspace/Minecraft-view-dataset/offline_evaluation/operations/goal006a/resume.py --plan /root/minecraft-runs/goal006-remaining800-20261014T101309Z-v2/plan.json --resume
```

ローカル状態だけの確認は末尾の`--resume`を`--preview`に置き換える。保存だけの再検証は同じスクリプトの`--verify-checkpoint`と`--release-id`を利用し、将来の未完成Draftを保存する場合だけ`--allow-create`を明示する。後者も有料生成を呼ばない。

**当日料金確認と不変条件**

UTC日付を実測し、2026-10-15 11:13:29–11:14:51 UTCに8件の公式文書を取得・確認した。固定モデル、料金、context・output上限、schema経路は凍結planと一致した。モデル照会・token-count・生成APIは呼んでいない。[公式確認記録と出典](pricing_refresh_20261015T111329Z/README.md)

既存`continuation_v1.plan.refresh_pricing`で同じplanの`pricing-revisions/0001.json`へ追記した。元plan・承認・authority・台帳のhashは変えていない。追記後は両providerとも`verified_on=2026-10-15`となり、当日の料金確認を通過した。日付が変われば既存の送信直前チェックが再び確認を要求する。[追記と原本不変の証拠](pricing_append.json)

canonical plan SHA256は`db78ec8e59b5921e248af42a7c8390647bd549c09c11e055c144729a247a2dce`、identityは`db5533bffb583d0b1efcbcdb7c1563edf2ac87606334e6ce6cd766c650075843`を維持している。既存原本1,275ファイルと凍結コード39ファイルのhash不変を確認した。固定順、モデル、prompt、schema、出力32768、retry0、累積USD30、親消費の扱いは変更していない。完了済み160件の全再評価や200GT・旧1200出力の再評価は行っていない。

**CPUテストと保存**

異なるテストケースは31件、すべてPASS。archive 26件とruntime 5件で、fail/error/skipは各0。変更後に必要な一部ケースを再実行した記録も残し、重複実行をテスト件数へ加算していない。

| 証跡 | ケース数 | 内容 |
|---|---:|---|
| [archive基本](archive_tests.junit.xml) | 16 | ID取得、ページネーション、既存asset再利用、失敗診断など |
| [archive復元](archive_provenance_tests.junit.xml) | 10 | 原本・台帳・予約・manifest・scopeの対応 |
| [archive修正後](archive_gh_fix_tests.junit.xml) | 3（再実行） | ダウンロードで変わるcounterなどを同一性比較から除外 |
| [archive送信長](archive_content_length_tests.junit.xml) | 2（再実行） | asset upload時のリクエスト構成 |
| [runtime](runtime_tests.junit.xml) | 5 | 凍結runnerの320→480、保存失敗で停止、次回再送0、未承認・不明usage拒否、preview |
| [archive asset IDゲート](runtime_asset_id_tests.junit.xml) | 1（再実行） | 初期320件の固定asset ID照合を加えた後の再開fixture |

実装ファイルは`archive.py`、`resume.py`、`tests/test_archive.py`、`test_resume.py`。実際の研究APIへ送信せず、fixture transportで再開・保存経路を検証した。

専用ブランチへの修正・証跡commitとremote一致の最終記録：**PENDING（rootが実際のpush後に更新）**。main・OS・SSH・ユーザー・Codex権限設定は変更しない。
