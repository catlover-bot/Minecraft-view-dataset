# Goal005 — completed16 preserved, fixed evaluation batch1 pending

Goal005の追加研究API送信は**0**。完了済み再スモークを結果として確定・保存し、既存固定60建築を維持した本評価960入力と、第1バッチ10建築160入力の未承認計画・ランチャーを用意した。測定の重大なバグは見つからず、残るモデル失敗を修正せず保持する。

root接続、実効workspace-write / auto_reviewを維持した。開始HEADは `e8c9d7e08880fcafc55aef812e959b3ebbddcc08`。過去SHAへのreset・再clone・OS/SSH/ユーザー変更は行っていない。OS隔離を保証する主張はしない。

## 確定した実測結果

既存run/lineageの全ロックを既存ファイルのread-only openで非待機取得し、COMPLETE_FIXED16、保留予約0を確認した。16入力・16raw・16評価行、natural stop/JSON/schema各16、executor/非空成功14、validation failure2、空出力0。全14metricを各1回CPUで再計算し元objectと完全一致した。実際の2 E入力の独立CPU復号も位置とexact material IDがGTに一致した。診断は4.76秒。

累積は **30 calls / retry4 / 354088 tokens / USD0.68679325 upper bound**。内訳は旧14/173509/USD0.25857425と、新16/180579/USD0.428219。旧消費を二重加算せず、不明usage0。請求確定額とは呼ばない。元plan file SHAは `c1fbd060b0d7459ad26d74b27994f9f2b891264a21d4835bf81bec32679c1030`、新16 ledger SHAは `10222aad7e3156f0cbd4603f650a7fd3bc85bf65b8a3fd30b42dda895809edb1`、旧14 ledger SHAは `6aaca585baf7a8dac2c66284f23eb110b2913ce0bad7119f1aaf5cd7044468c0`。原本・authority・制御ファイルの基準SHAを [originals_before.json](originals_before.json) に保持した。

OpenAIの2失敗はbuilding060のO/Jで、0始まりoperation index5/2（6番目/3番目）の `block="quartz (white)"` が最初の違反。空白・括弧を許さない材質ID構文に違反する。schemaは文字列構造を検査し、ローカルValidatorはID構文を検査するため矛盾しない。Validator緩和、GT補修、失敗行削除を行っていない。[全16行CSV](results/all16.csv)、[失敗の規則・操作抜粋・raw識別子](results/failures.json)。

|Provider E|建築|occupancy exact|labelled exact|add / delete / replace|主な不一致|
|---|---|---|---|---|---|
|OpenAI|060|true|false|0 / 0 / 637|nether_brick→glowstone392、stone_slab→stonebrick239、glass→brick_block3、glowstone→glass3|
|OpenAI|075|true|false|0 / 0 / 149|stone_slab→stone149|
|Anthropic|060|false|false|31 / 0 / 4|fillを単一setへ縮小、欠落、境界の拡張・上書き不足、材質転記差|
|Anthropic|075|false|false|298 / 0 / 0|50 cuboidのpalette0=cobblestoneをairへ転記|

addは予測に不足するvoxel。4件ともaligned shiftは0。モデル内部の原因やalias同値性を推測で確定せず、exact ID規約を維持する。[全混同行列・不一致cuboid](results/e_differences.json)、[実操作の境界・重複・palette転記](results/e_transcription_details.json)、[実voxelの投影PNG](projections/README.md)。

O/J common-valid建築はOpenAI075、Anthropic060/075。valid-onlyと識別済み完了応答のfailure0 bounded IoUを併記し、失敗MAE/repairはnull。NOT_RUNや測定不能を0にしない。[結果解釈と条件別表](results/README.md)。2建築smokeのp値・一般的優劣・言語化の因果的情報損失は主張しない。旧smokeとはprompt/schema/E/capが異なり、経緯としてのみ記録する。

## 実測成果物を先にGitHub保存

診断・報告commit `da2be613a820f6261739d7b05af6e9b4801ce823` を専用ブランチへpushしremote SHA照合後、新規[未公開結果Draft412128399](https://github.com/catlover-bot/Minecraft-view-dataset/releases/tag/untagged-af589f3f57d96f8f8c3a)へ保存した。結果archive asset **638311327**、805523 bytes、SHA256 **b1a615cb1390cbf3e62579f2f4a6ea76ae1be8a2a5df019b8726ae69439f5422**。全6assetsのGitHub digest・再取得SHAが一致し、93ファイルの安全展開が成功した。

復元先でraw16・raw→parsed→voxel14・2失敗を再検証し、metric探索は繰り返さなかった。保存planのcanonicalパスを保持し、コピー先からの有効化は拒否された。[再取得・復元証跡](results_github_verification.json)。旧source asset637059866 / SHA97b5d9f0862449c881cd5804485f1a27cc5164baf3f51df09c66c2658ac3d6da、および旧14 raw asset637707690 / SHA8f60b4df66f11cec77093d04c8f077652bc65cf42219ffbcf2161d117481ec8eは既存IDで参照し、再圧縮・重複uploadしなかった。結果保存を完了してから実際の新disabled planを作成した。

## 固定60と第1バッチ

Goal002の10smoke/60evaluation/30restとsource hashを再利用し、60を再抽出していない。060/075を本評価へ混ぜていない。既存データの固定評価部分集合であり未見データとは呼ばない。O/Jは保存OpenAI/Anthropic/Gemini説明を各建築で使用し、P/Eは各1回。各provider O180+J180+P60+E60=480、両provider960。現行prompt/schema/Validator/Executor/metric、E cuboid、モデル2種、output32768を保持した。[固定sourceと全960 input hash](planning/manifest.json)、[CSV索引](planning/input_manifest960.csv)。

既存stratum_draw_orderで各stratumの2件を順に選んだ第1バッチは **087,026,007,032,092,078,045,023,055,062**。各stratum2建築、各建築16入力で160。provider交互、条件順は事前固定で建築ごと回転する。結果に依存する選抜はない。残り50建築800入力は別承認待ち。第1バッチ後に条件を変えた場合は同条件としてpoolしない。

60件だけをCPU診断し、参照E復号/Executorは全件完全一致。既存tiktoken0.14.0 / o200k_baseとcacheだけを使用し、token API/依存取得0。9/60の参照出力本文がcap32768を超え、第1バッチの032/092も含む（最大79978 local tokens）。対象除外・入力切詰め・cap変更は行わず、現条件の資源制約として記録した。参照は最短列ではなく、実出力長・Anthropic正確token数・完了を保証しない。[CPU長診断と条件変更案との差](planning/README.md)。

|計画|条件重み付き費用外挿USD|時間外挿|全context+output予約USD|
|---|---:|---:|---:|
|160|3.1722475|約2.18時間|42.350080|
|960|19.033485|約13.09時間|254.100480|

16実応答のprovider×条件別平均へO3/J3/P1/E1を掛けた外挿。2建築のみで、O/J説明はOpenAI由来しか測定されていないため、建築規模・他説明モデル・出力長・待ち時間の不確実性が大きい。[実測明細・費用表](planning/costs.json)。料金・context・structured outputは実UTC **2026-10-13T21:24:22.644662Z** に公式文書を取得して確認した。[公式根拠とURL](official_support.md)。

## 未承認planと実在ランチャー

本評価実装commit `dd4925257cbbdfb88ac343c6c8d29483a69fa364`。旧hash-bound resmoke/server_v2は変更せず、新しいbatch_v1でscope/台帳/順序/料金再確認のみを拡張した。レビューで新モジュールのdisabled CPU evaluateによる空enabled作成を修正し、送信機構の実装バグとモデル失敗を区別した。

Canonical planは `/root/minecraft-runs/goal005-batch1-20261013T213420Z/plan.json`、SHA256 `231b6daffb671511cd3adc22847d9d72c26a37db14b5c6793520a6f987001b13`。`api_allowed=false`、enabled/authority/予約は未作成。新旧累積USD10/calls190/retry4を提案するが未承認。完了費用予測ではなく、次の1送信分の保守的予約が残額へ収まらなければ未送信で停止する。[未承認plan](pending_plan.json)、[prepare証跡](pending_preparation.json)、[実preview検証](batch_preview.json)。

利用者のキーのあるターミナルで使う1本の開始コマンド、実確認文字列、同一ledger再開、翌UTC日の公式再確認を追記sidecarとして保存する手順は [OPERATIONS.md](OPERATIONS.md)。Codexは実確認語を入力せず、previewだけをsocket接続禁止・キー不要で実行しexit0を確認した。未知usage/重大運用障害では停止し、自動retry・fallback・予算初期化はしない。

未承認計画は別の[未公開準備Draft412133442](https://github.com/catlover-bot/Minecraft-view-dataset/releases/tag/untagged-91b63f2474552cffbd38)へ保存した。準備archive asset **638334591**、1163923 bytes、SHA256 **e68f9039fa0661dcdb5b35bc5726d417c77b0fc2ff03519a7c0964bdb2a0bed0**。全6assetsのGitHub digest・再取得SHAが一致し、20ファイルの安全展開と、復元plan/manifestのcanonical原本とのbyte一致を確認した。コピー先のplanは起動拒否され、元planのenabled/authorityは未作成のままである。[準備の再取得証跡](pending_github_verification.json)、[実測結果と未承認計画を分けた保存索引](checkpoint_index.json)。

## 検証と残る範囲

関連既存229、診断14、入力14、バッチ25の合計**282個の異なるテストがPASS**。fail/error/skip0。新規ガード修正に関係する3ケースだけを追加実行し、テスト件数を増やすための全再実行はしていない。[テスト集計](test_summary.json)。旧6 collection errorsは [既存証跡](../goal003/legacy_suite_reference.json) を保持し、一括skipやダミー修正を行っていない。200GT/旧1200の全再評価・環境再構築・新依存取得は0。

本Goalを妨げる重大な測定BLOCKEDはない。残るモデル失敗、9件の参照出力長超過、将来の有料実行承認は結果・制約・承認待ちとして明示している。本評価はNOT_RUNであり、未承認計画の完成を実験完了とは呼ばない。

[最終ローカル監査](completion_audit.json) は原本・制御ファイル、旧hash-boundコード、固定source/plan、テスト証跡、両Draftの再取得検証を照合した。報告だけの最終commit前HEADを記録し、最終push後のremote SHA一致とmain不変は完了報告に示す。
