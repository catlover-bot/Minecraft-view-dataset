# Goal001C server-only checkpoint

実行環境は88.216.191.99 / root / Ubuntu24.04 / `/root/workspace/Minecraft-view-dataset`。開始HEADは`7309327efd1c19dda793b19e8d8e6ce0b5868db9`、専用ブランチは`research/server-only-20261013`。既存SSH、mhirotaka、OS設定は変更していない。セッションは開始時からsandbox disabled/approval neverであり、このGoalで変更していない。OSレベルの隔離や保護有効性は保証しない。

GitHubのmain、到達可能な全25コミット、remote branch、タグ、Releases、Actions artifactメタデータ、LFS/submodule宣言を調査した。partial promisor cloneであることを確認し、ツリーを調べて欠落blobと欠落パスを区別した。原本17339ファイルをrepo/ref/path/blob/SHA256で索引化した。無関係なrepoや手元PCは探索していない。

主要コホートはGT200、対応画像2400、説明500（OpenAI200、Claude200、Gemini-v4 100）。v1 Gemini100、ARRのcanonical入力、attempt、成功actions、prediction voxel、Direct100の生出力、反復30の生run、旧fixed_builder/executor/metricsはMISSING。集計CSVの照合と生成出力からの再現を区別する。追加LLM-authored20建築は別コホートとして数え、主要200に加えていない。

旧ARRのhistorical_replayはBLOCKED。今回のserver_v2と実測指標はnew_implementation。HTTP mockのテストはfixture_only。旧API互換性・ARR数値同等性は確認できず、ダミーimportや一括skipを追加していない。古いi2t2b-v5保存出力1200は別版として再評価した。

全GT200でExact-spec→voxelの完全一致200/200、GT→actions→executor完全一致200/200、既知shift(1,-1,1)の厳密探索による回復200/200。これらはLLM成功率に含めない。CPU34.432秒、最大RSS58384KiB。再評価1200中1168成功、32不適合。32件は13建築の曖昧な材質ラベルで、2件にはU32切詰めを確認した。原本を補修せずnull指標とし、履歴上のMinecraft実行失敗とは断定しない。再評価112.001秒、最大RSS64328KiB。

新テスト91pass / 0fail / 0skip / 0error（pytest1.88秒、実時間と最大RSSはnew_tests_summary.jsonの/usr/bin/time実測）。旧テストは0pass / 0fail / 0skip / 6collection error（欠落import、pytest0.64秒）。専用venvを新規作成し、全依存をrequirements.lockへ固定、pip check成功。worker2、BLAS/OpenMP1。空き容量はGT検証時91721351168bytes。

固定subsetはseed20261013、v1/v4×GT規模五分位からsmoke10/evaluation60、互いに重複なし。通常評価予定はOriginal150+新しいlossless Canonicalized150=300件（当初360から欠損60を除外）。Partial Oracle60、Exact-spec Oracle60は建築ごとに一度。smoke通常50+Oracle20=70、評価通常300+Oracle120=420。既存データの部分集合であり未見データではない。欠落説明の新規生成には別途承認した新コホートが必要。

OpenAI Responses/Anthropic Messages adapter、明示モデル、共通入力/action schema/validator/executor/metrics、永続予算予約、atomic write、lock、attempt ID/UTC、commit/input/prompt/config/model/output hash、raw/parsed/voxel/metrics分離、full-identity resume、失敗分類、paired差と建築cluster CI、Holm utilityを実装・テストした。API仕様は公式docsを2026-10-13に確認。モデルavailabilityはNOT_RUNで未確認。

研究APIはNOT_RUN、送信0回。実API成功は主張しない。初回生成には環境中のANTHROPIC_API_KEY（第二Builder）またはOPENAI_API_KEY、api_allowed=true、--allow-api、明示provider/model、呼出回数/token/USD/attempt/output上限、確認日付き公式の価格・context上限・long-context割増を含む上限料金が必要。不明usageは0扱いせず予約を保持して停止する。

保存対象はコード・small summaries・原本manifestと、完了CPUタスクのallowlisted snapshot。GitHub保存・再取得検証の結果はcheckpoint_index.jsonへ記録する。古いARR原本が欠けるためcheckpoint全体はPARTIALで、旧研究の完全復元可能とは報告しない。大きい原本は一度だけsource assetへ保存し、将来snapshotはsource SHAで参照する。

実在するCLIと将来の生成/checkpoint/restore手順は`offline_evaluation/server_v2/README.md`。大きな圧縮処理とアップロード計測は完了後にindexへ追加する。サーバー側commitだけではバックアップ完了とは呼ばない。

| Saved i2t2b condition | Success | Invalid | Absolute IoU mean | Aligned IoU mean (radius1) |
|---|---:|---:|---:|---:|
| rebuild_world_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001 | 200 | 0 | 0.123234 | 0.143879 |
| rebuild_world_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_common_v8_struct_self_refine_no_gt_tuned | 200 | 0 | 0.130483 | 0.150071 |
| rebuild_world_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned | 200 | 0 | 0.154415 | 0.178367 |
| rebuild_world_schema_material_v5_repair_openai_gpt_5_mini | 193 | 7 | 0.124885 | 0.149274 |
| rebuild_world_schema_material_v5_repair_openai_gpt_5_mini_common_v8_struct_self_refine_no_gt_tuned | 187 | 13 | 0.122143 | 0.150953 |
| rebuild_world_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned | 188 | 12 | 0.157093 | 0.184206 |

These are new metrics on saved older artifacts; not reproduced ARR paper values. Descriptive building-cluster CIs are in summary.json. No significance claims are made.

Official sources: [Codex security](https://developers.openai.com/codex/agent-approvals-security), [Ubuntu admin rights](https://help.ubuntu.com/stable/ubuntu-help/user-admin-explain.html), [gh auth login](https://cli.github.com/manual/gh_auth_login), [gh auth setup-git](https://cli.github.com/manual/gh_auth_setup-git), [gh release create](https://cli.github.com/manual/gh_release_create), [GitHub release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases), [GitHub large files](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github), [OpenAI text generation](https://developers.openai.com/api/docs/guides/text), [Anthropic Messages](https://platform.claude.com/docs/en/api/messages/create). Documentation queries are distinct from paid research generation. Ubuntu URL could not be fetched by the web tool; no OS changes were made.

GitHub保存・再取得・安全な復元の検証が完了した。未公開Draft Release、9 assets、全hash照合、計測値、次のCLIは[checkpoint_index.md](checkpoint_index.md) / checkpoint_index.jsonに記録。実装確定commitは`6bf1629cc4a0bca92d6890389148ae38c4b1a3ee`、索引はその後続commitで保存する。完全性はPARTIALのまま。
