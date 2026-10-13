# Goal002 checkpoint索引

新Draft **411807868**（未公開）、tag `checkpoint-goal002-20261013-141854`、target commit `2d21167170c96a93e7b5d5bc7fbff532838c7661`。専用branchへのpushとremote SHA照合済み。詳細asset ID・URL・SHAは [JSON索引](checkpoint_index.json)、再取得・復元証跡は [github_verification.json](github_verification.json)。最終報告commitの正確なSHAは、Draftの `final_repository_state.json` と最終応答に記録する。

[GitHub Draft](https://github.com/catlover-bot/Minecraft-view-dataset/releases/tag/untagged-e3968d3f7252cd13c52f) には、checkpoint archive・FILES manifest・metadata・SHA256SUMS・RESTORE・Release notes・既存source参照の7 assetsと、復元検証・文書訂正・代表CPU検証スクリプトの3 assetsを保存し、10 assets全件のGitHub digest/local SHA/再取得SHAを照合した。最後にrepository state receiptを別assetとして追加する。archive SHA256は `5900ecb4ef05539fd2cdfeccce26f9a960f20f7adc20b66c0b3ade538baac627`。

4307完了済みCPU/準備ファイルを凍結し、snapshot4387ファイルを新規ディレクトリへ安全に復元した。旧source asset637059866（旧Draft411682582）も新しく再取得し、SHA256 `97b5d9f0862449c881cd5804485f1a27cc5164baf3f51df09c66c2658ac3d6da` と全11029ファイルのSHAを検証した。sourceの再圧縮・再アップロードは行っていない。

復元したコードでv4全100建築のcohort/sourceを再監査し、evaluation480/initial smoke8/full smoke80の各入力manifestが一致した。代表2GTでExact-spec往復/Executor/既知shift6試験/遠い原点がPASS。旧材質問題の代表例はgeometry有効・材質UNKNOWNのままで、原本SHA一致。4つのdisabled Builder dry-runと960/16計画行のNOT_RUN/null集計もPASS。代表復元検証37.20秒、最大RSS93064KiB。全200/1200の二度目の復元CPU再評価とは呼ばない。

新v4の必要sourceと準備物は **COMPLETE_VERIFIED**。保存hash検証は **VERIFIED**。有料実験は **NOT_RUN・送信0回**。旧研究全体は原本ARR欠損のため **PARTIAL / HISTORICAL_REPLAY_BLOCKED**。キー存在boolはOpenAI=false/Anthropic=false、予算承認なし。

既存25報告、旧Draft411682582と全asset ID/hash、mainを保全した。新規追加asset以外は上書きしていない。Jの再シリアライズはキー順・空白に加え同値escape/数値の表記を変える場合があるという [文書訂正](documentation_errata.md) を伴う。型付き木の値・型は保存され、実装や数値結果を変更していない。

保存先 `/root/minecraft-checkpoints/goal002-20261013-141854`、復元project同ディレクトリの `reconstructed_project/`、復元run `/root/minecraft-runs/goal002-restored-20261013-141854`。元worktree/runへの上書きなし。復元後空き容量74.86GiB。sandbox起動エラーなし。実効workspace-write/auto_reviewを使用し、OS隔離は保証しない。
