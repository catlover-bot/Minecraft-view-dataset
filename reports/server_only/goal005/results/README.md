# Goal005 saved-response CPU diagnosis

追加研究API送信は0。Python 3.12.3、CPU thread環境値1で完了済み16応答だけを読取り、14有効出力の全metric objectを各1回再計算して元値と完全一致した。2建築の実E入力を独立デコードし、GTの位置・exact material IDと完全一致した。実験中の133原本ファイルのSHAは診断前後で不変。診断4.759972519秒、tokenizer未使用。

16入力の全行は `all16.json` / `all16.csv`。response/natural stop/JSON/schemaは各16、executor成功・非空再建築は14、空出力0、validation failure2。失敗のMAE/repairはnull（CSV空欄）。識別された完了応答の失敗0は4種類のbounded IoUにだけ適用し、NOT_RUN・運用上測定不能を0にしない。

OpenAIの2失敗は両方building_060。Oのoperation index 5（6番目）、Jのindex 2（3番目）の `block="quartz (white)"` が最初の違反。空白と括弧を含むため、凍結coreのlower-case material ID構文を満たさない。schemaは文字列の形だけを確認し、局所validatorがID構文を確認する分離仕様のため、schema_successとlocal failureは矛盾しない。座標・operation数・touch/live voxel制限による失敗ではない。原文からのモデル出力の仕様違反として保持する。Validator実装バグや主指標変更を必要とする証拠は見つからず、失敗を成功へ書き換えていない。詳細識別子・raw SHA・安全な操作抜粋・規則は `failures.json`。

全16の凍結input/prompt/schemaから再構成したrequest payload hashはledgerと一致。rawのID/model/usage/natural stopも一致し、有効14件のraw→schema→local validation→parsed→executed voxelは完全一致。`request_chains.json` と `metric_replay.json` に保存。これは実際の送信時ledgerに束縛されたpayloadとの照合であり、外部provider側の別記録を取得したものではない。

| Provider | Scene | Occupancy exact | Labelled exact | GT / pred voxels | add / delete / replace | aligned material IoU |
|---|---|---|---|---|---|---|
| openai | 060 | True | False | 2554 / 2554 | 0 / 0 / 637 | 0.6007521153243497 |
| anthropic | 060 | False | False | 2554 / 2523 | 31 / 0 / 4 | 0.9847537138389366 |
| openai | 075 | True | False | 8018 / 8018 | 0 / 0 / 149 | 0.9635116934002694 |
| anthropic | 075 | False | False | 8018 / 7720 | 298 / 0 / 0 | 0.9628336243452232 |

addはpredに不足しGTへ追加が必要な数、deleteはpredの余分なvoxel、replaceは同位置のexact ID不一致。4 E応答のaligned shiftはすべて[0,0,0]。

OpenAI060は同じ334 cuboid境界を保持したまま70 cuboidで材質を取り違えている。nether_brick→glowstone392 voxel（27 cuboid）、stone_slab→stonebrick239（37）、glass→brick_block3（3）、glowstone→glass3（3）。OpenAI075は657境界を保持し、stone_slab→stone149 voxel（63 cuboid）。palette外IDや境界・重複・欠落はない。入力palette indexに対応しない材質への転記は確認できるが、モデル内部の原因までは推測しない。旧版/新版aliasとして同一視する根拠もなく、任意mappingを適用していない。

Anthropic060は31 voxel不足：stone_slab18、brick_block6、stonebrick6、glass1。複数voxelのfillを1セルのsetに縮めた例は入力cuboid index152 [1,9,9,1,10,9,7]に対して出力operation154がset(1,9,9)だけ。glass1のcuboidは欠落。2置換はglowstone→nether_brick、別の2置換はbrick_blockのx2を22から23へ広げた後で、対象stonebrickのy=9だけをsetし、y=10を上書きせず残した結果。出力に4重複訪問座標あり（glassの同材質重複2と、後から書き直されたstonebrick座標2）。詳細の元bounds・実operation・重なりを `e_transcription_details.json` に保存。

Anthropic075は50 cuboidのpalette index0=cobblestoneをairに転記して298 voxel不足。657の境界は保つが、airは非air palette外であり、指定された建築の完全変換ではない。`e_differences.json` は全GT material→pred material混同行列（不足をair列と表記）、件数、座標例、全不一致cuboidを含む。

OpenAI O/Jのcommon-validはbuilding_075のみ、Anthropicは060と075。先にIDの共通部分を確認した。同じ有効nだけからpaired差を作っていない。

| Provider | Condition | valid n | valid-only aligned occupancy IoU | failure0 n | failure0 aligned occupancy IoU |
|---|---|---|---|---|---|
| openai | O | 1 | 0.19619629483642098 | 2 | 0.09809814741821049 |
| openai | J | 1 | 0.29105989005690036 | 2 | 0.14552994502845018 |
| openai | P | 2 | 0.15615641914445041 | 2 | 0.15615641914445041 |
| openai | E | 2 | 1.0 | 2 | 1.0 |
| anthropic | O | 2 | 0.2633832719191372 | 2 | 0.2633832719191372 |
| anthropic | J | 2 | 0.29619085605192974 | 2 | 0.29619085605192974 |
| anthropic | P | 2 | 0.20494981369424975 | 2 | 0.20494981369424975 |
| anthropic | E | 2 | 0.9753479006612569 | 2 | 0.9753479006612569 |

この2建築smokeにはp値・一般的優劣・因果的改善の結論を付けない。Eは完全な離散仕様の変換診断、PはGT属性だけの不完全仕様で、自然言語の情報損失そのものではない。旧smokeはprompt/schema/E/capが異なる経緯としてのみ扱う。残るモデル失敗は本評価準備を止める計測バグではない。

診断関数は旧hash-boundコードを一切編集せず再利用し、実行前にplanのimplementation hashを確認する。新しい出力先だけへ書込む再現入口：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-server-v2-20261013/bin/python -B -m offline_evaluation.goal005.diagnosis --plan /root/minecraft-runs/goal004a-ttyfix-20261013T194600Z/plan.json --output /root/minecraft-runs/goal005-new-unique-diagnosis
```

保存済みarchiveを診断専用に復元した後は `diagnosis.verify_archive(restored_root)` が、original/plan_root/の16raw・14 parsed/voxel・2失敗・payload・authority/approvalを読取り照合する。元plan中のcanonicalパスを変更せず、activation/予約は行わず、metric探索を繰り返さない。旧親runの大容量assetはroot checkpoint索引の既存ID/SHA参照で保全する。

関連fixtureは14 tests PASS（0.33秒）。局所規則の先行順、独立cuboidデコード、exact-ID差、異なるcommon-valid集合、failure0除外、非上書き、復元読取りとraw/request/control破損拒否を確認。`tests.junit.xml` を参照。
