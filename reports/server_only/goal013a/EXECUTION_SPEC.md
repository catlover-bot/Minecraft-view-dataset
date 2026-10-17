**Goal013A 実行前固定**

方法はMETHOD_SPEC.md、関連分析はASSOCIATION_SPEC.mdと旧Goal013 SCORING_PROTOCOL.mdに固定する。`SCORER_CONFIG.json` SHA256 `589ed76a43061d966f2cf99ebbb0721f9331bcc1ea0f94f5119a39a523d2670c` が実ローカルパス、全source/weight実測hash、依存版、許容差、60scene frameを束縛する。すべてpost-hoc探索で、preregistrationではない。

新tools root: `/root/minecraft-tools/goal013a-20261017T131846Z`。新run root: `/root/minecraft-runs/goal013a-20261017T131846Z`。推論前にcode commitと別spec commitをCLIへ40桁で渡し、祖先関係・各commitの全対象filesと現在bytes一致を検証する。実測checkpointは353,976,522 bytes、SHA256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`。取得済み全bytesから再計算した値で公式expectedと一致した。transform JSON hashは `17a91b02b48b7ffc5ed0e421e7d5f5265631c1c4c2589c1e7c4bf37419247907`。

CPU wheelは公式2.9.1/0.24.1対応ペア、Python3.12 x86_64用。numpy1.26.4、Pillow11.3.0、ftfy6.3.1、regex2025.9.18、packaging25.0等22wheelをrequirements.lockの全hashでoffline導入した。全installed版とartifact/license hashはDEPENDENCY_MANIFEST.json・ENVIRONMENT_RECEIPT.json。bootstrap pip24.0はPython ensurepip付属。新venv内以外のpip installは行わない。推論前import smokeではtorch/torchvisionを通信禁止guard下でimportでき、モデルload・推論は0だった。

公開取得は57 HTTP requests、584,832,575 response body bytes。redirect0、HTTP200が57件。これはTLS/header/wire overheadや文献閲覧のブラウザ通信を測った値ではない。研究API送信は別に0。3回のローカルresolver実装エラー後は取得済みbytesを再hashして再利用し、重み再downloadなし。過去の取得禁止は今回の明示許可で専用新規領域に限り変更され、その経緯をSTART_STATE.jsonに残した。

**固定参照例とpilot順序**

参照例はscene/provider辞書順の先頭 `buildings_100_v4_building_001` / `anthropic`、summary SHA `38f6d4b4b07dd02fdc4bd365b03acc318d0de34a46aa3a9dd06ac01a092dd55b`。その原summaryを6回反復し、保存request順の6画像と組み合わせる。source loaderが原文と全画像bytesをhash再検証する。原著者ASTの実関数を動かし、token exact、tensor exact、normalized featureとqはatol=rtol=1e-5。別float64 scalar検算、batch1/8、cache/replay、6値逆順のmean不変を確認する。得点大小は判定に使わない。

parity後にfeature cacheとcache countersを空にする。scene ID順先頭1件をfresh採点し、その結果を保ったまま先頭10件のnested pilotへ進む。10件pilot経過を6倍した基礎推計に対して、残50件の追加時間へ安全倍率2を掛ける: current cumulative CPU/wall + 2×10件採点時間×5。これがCPU3600秒/wall7200秒未満なら同プロセスで残50件を採点する。parity/model loadの使用量もcurrentへ含む。pilotはBuilder結果を結合しない。

batch最大8、workers0、OMP/BLAS/MKL/NUMEXPR/torch intra/inter各1。modelCPU/float32/eval/jitFalse/inference_mode。固定CPU上限3600秒、wall7200秒を累積管理し、RSS目標4GiB、RSS/address space上限6GiB。開始直前にhost/cgroup利用可能量から2GiB余裕を引き6GiBが適合することを再確認する。POSIX CPU signalの整数秒切上げと、計算中断後のsource hash照合・partial receipt保存専用のhard-limit120秒余裕を区別する。余裕は追加推論の許可ではない。新領域8GiB以下/空き10GiB以上を各sceneで確認する。Python audit hook/read-only locksはOS隔離保証ではない。

推論中はsocket通信と任意subprocessを禁止し、source inventory用read-only gitのみ許可。書込みは新stage出力・新tmp/cacheだけ。TMPDIR/XDG_CACHE_HOME/TORCH_HOME/PIP_CACHE_DIRを専用rootへ固定して照合。scoreは各行をexclusive保存し、停止時に計算済み/残件を残す。出力既存時の上書き再実行は拒否する。

**出力とPhase 2境界**

score/REFERENCE_PARITY.json、PILOT_01.json、PILOT_10.json、SCORING_SUMMARY.json、SCORES_180_CANONICAL.json、SCORES_180.json、TOKEN_REPRESENTATIVENESS.json、CROP_METADATA_360.jsonを新canonical runへ保存。77実token IDsと固定3crop PNGはcanonicalだけ、public scoreではtoken IDsを除きhash/count/maskを保持する。crop例は固定building001/003/005のfirst request画像のみ。生summaryやGT/full raw responseをGitへ複製しない。

全旧source・既存環境の終了hash確認がPASSしてからscore/SCORE_FREEZE.jsonをexclusive保存し、code/spec/config/parity/score/source/coverage/実行receiptをhashで束縛する。Phase2は別CLI stageでfreeze全bytesSHAを引数渡しして検証し、全体/各provider代表性PASS後だけ旧Goal007の固定O結果を開く。代表性gate不通過ならPHASE2_NOT_RUN.jsonを保存して停止し、相関・CI・実association Nをnullとする。採点成功と尺度妥当性を混同せず、旧RQ2は変更しない。

実行CLI（code/spec SHAは各commit完了後に記録）:

```sh
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 TMPDIR=/root/minecraft-tools/goal013a-20261017T131846Z/tmp XDG_CACHE_HOME=/root/minecraft-tools/goal013a-20261017T131846Z/cache TORCH_HOME=/root/minecraft-tools/goal013a-20261017T131846Z/cache/torch PIP_CACHE_DIR=/root/minecraft-tools/goal013a-20261017T131846Z/cache/pip /root/minecraft-tools/goal013a-20261017T131846Z/venv/bin/python -B -m offline_evaluation.analysis.goal013a.run_pipeline score --config /root/workspace/Minecraft-view-dataset/reports/server_only/goal013a/SCORER_CONFIG.json --code-commit CODE_SHA --spec-commit SPEC_SHA
```

同じrootの再上書きは不可。必要な再実行は新しい専用rootとその実測manifestを別途固定し、累積download/inference使用量を引き継ぐ。実行後に都合よくcheckpoint/prefix/許容差/token gateを変更しない。

実推論前コードcommit: `6cbc854f61b57accdd83a9a59fcfd0ddfd00f480`。仕様commitは本ファイルを保存する次の別commitで、実行引数とfreeze receiptに実SHAを記録する。新規合成fixture148件PASS（fail/error/skip0、3.71秒）、FIXTURES.junit.xmlとFIXTURE_RESOURCES.txtを同仕様commitに保存。
