# Goal002 運用手順

研究APIは **NOT_RUN・送信0回**。有料smokeの利用者承認はまだない。以下の有料開始・承認・保存コマンドは将来用であり、今回実行していない。固定対象、入力・prompt・config・実装hash、予算、保存先を確認してから承認する。比較条件は [protocol.md](protocol.md)、公式確認記録は [official_pricing.json](official_pricing.json) を参照。

作業ディレクトリは `/root/workspace/Minecraft-view-dataset`。既存venvを使用し、CPU処理はworker2・BLAS/OpenMP1を基準とする。

```bash
P=/root/workspace/Minecraft-view-dataset/.venv-server-v2-20261013/bin/python
R=/root/minecraft-runs/goal002-20261013-141854
S="$R/smoke_proposal"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

disabled準備物は次の新規ディレクトリに保存する。既存準備物の再作成・その場での `api_allowed` 反転は行わない。

| ディレクトリ（`$R/` 以下） | 計画入力数 | 用途 |
|---|---:|---|
| `builder_openai_initial_smoke` | 8 | 2建築×O/J/P/E、OpenAI |
| `builder_anthropic_initial_smoke` | 8 | 同じ対象・入力、Anthropic |
| `builder_openai_evaluation` | 480 | O180＋J180＋P60＋E60、本評価準備のみ |
| `builder_anthropic_evaluation` | 480 | 同じ内訳、本評価準備のみ |
| `smoke_proposal` | 16 | 両provider合算の承認待ち提案 |

次のdry-runはAPIを送信しない。今回すでに実行した、freeze前の確認コマンドである。凍結後は保存済みdry-runログを参照する。キー設定後の再確認は新enabled runで行う。モデル・アカウントの実可用性を確認するものではない。

```bash
$P -B -m offline_evaluation.server_v2.cli dry-run --run-dir "$R/builder_openai_initial_smoke"
$P -B -m offline_evaluation.server_v2.cli dry-run --run-dir "$R/builder_anthropic_initial_smoke"
$P -B -m offline_evaluation.server_v2.cli dry-run --run-dir "$R/builder_openai_evaluation"
$P -B -m offline_evaluation.server_v2.cli dry-run --run-dir "$R/builder_anthropic_evaluation"
$P -B -m offline_evaluation.server_v2.cli smoke-dry-run --run-dir "$S"
```

smoke対象は固定v4 splitの `building_060` と `building_075`。O/Jには保存済みOpenAI説明を使い、P/Eは建築ごとに一度だけ与える。上限は両provider合算で送信20回、再試行4回、1入力2attempt、USD5、8,327,680 tokens、各出力16,384 tokens、concurrency1、隠れた再試行0。token上限は `20 × (400000 + 16384)` の保守的予約に対応する。USD5はアプリ側の停止上限で、実費見積や16入力完了の保証ではない。

2026-10-13の公式確認では、要求モデル `gpt-5-mini` はDeprecated、context400000・最大出力128000・入力USD0.25/百万tokens・出力USD2/百万tokens。公式に記載された返却snapshot `gpt-5-mini-2025-08-07` だけを明示許可し、送信モデルIDは変えない。`claude-haiku-4-5-20251001` はActive (legacy)、retirementは2026-10-15より前ではないとの記載、context200000・最大出力64000・入力USD1/百万tokens・出力USD5/百万tokens。実アカウントでの利用可否は双方 **NOT_RUN**。開始直前に公式価格・上限・提供状況を再確認する。価格確認の有効期間は7日で、廃止予定があるモデルの提供をその期間保証するものではない。利用不可でも別モデルへ置換しない。

OpenAIの出力上限にはreasoning tokensも含まれ、出力料金を適用する。Anthropic側はextended thinkingを有効化しない。全context＋出力の上限を送信前に永続予約する。不明usage・timeoutは予約を保持して両providerを停止し、0円扱いや自動再送をしない。成功入力の再送は禁止。承認済みディレクトリを複製して新しい予算を開始することもできない。

将来の承認ファイルを作る場合は、未使用の保存先を先に決める。下記の `FUTURE_UNIQUE_ID` を実際の一意なIDへ置換する。承認テンプレート作成は送信0回で、`manual_confirmed` はfalseのまま。

```bash
E=/root/minecraft-runs/goal002-approved-smoke-FUTURE_UNIQUE_ID
M=/root/minecraft-runs/goal002-manual-review-FUTURE_UNIQUE_ID
mkdir "$M"
A="$M/approval.json"
(
  set -o noclobber
  $P -B -m offline_evaluation.server_v2.cli smoke-approval-template \
    --run-dir "$S" --enabled-directory "$E" > "$A"
)
```

利用者が対象16入力、両モデル、全上限、canonicalな `$E` を明示承認した後だけ、利用者が `$A` の `manual_confirmed` をtrueへ変更する。他の項目は変更しない。今回はその承認を得ていない。価格・入力・prompt・config・実装を更新する必要がある場合は、新しいdisabled準備物と新しい承認を作る。APIキーは環境の `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` を使い、ファイル・コマンド例・ログに値を書かない。

承認後の将来手順。`approve-smoke` は新しいenabledディレクトリを作るだけで送信しない。両キー存在boolと保存先・上限を確認した後の `smoke-generate --allow-api` が有料送信を開始する。既存runや提案をenabledへ直接変更しない。

```bash
$P -B -m offline_evaluation.server_v2.cli approve-smoke \
  --run-dir "$S" --destination "$E" --approval "$A"
$P -B -m offline_evaluation.server_v2.cli smoke-dry-run --run-dir "$E"

# 有料送信：今回はNOT_RUN。上記の利用者承認後のみ。
$P -B -m offline_evaluation.server_v2.cli smoke-generate --run-dir "$E" --allow-api

# 以下はCPUのみ。同じoccupancy最適shiftを全Aligned指標に使用する。
$P -B -m offline_evaluation.server_v2.cli evaluate \
  --run-dir "$E/runs/openai" --alignment-mode strict_translation
$P -B -m offline_evaluation.server_v2.cli evaluate \
  --run-dir "$E/runs/anthropic" --alignment-mode strict_translation
$P -B -m offline_evaluation.server_v2.cli summarize-v4 --run-dir "$E" --smoke
```

停止後は保存されたledger・attempt・raw・出力を保持し、同じcanonical `$E` を使う。認証・課金・モデル不在・context・不明usage等の停止理由を確認する。ledgerを初期化したり、別run/providerへ移して予算をリセットしない。smoke集計は通信・実行確認で、有意差や一般化の結論を付けない。smokeから計画960件の本評価へ進む処理はない。

全CPU・生成処理が終了し、active taskがない状態でだけfreezeする。凍結したcompleted allowlistのファイルを書き換えない。将来の承認操作は提案に新しいactivation記録を追加し、実行状態は新enabled runへ保存する。新checkpointは前回のsource assetを参照し、source一式を再圧縮・再アップロードしない。

```bash
C=goal002-smoke-FUTURE_UNIQUE_ID
$P -B -m offline_evaluation.server_v2.cli freeze --run-dir "$E"
$P -B -m offline_evaluation.server_v2.cli checkpoint \
  --run-dir "$E" --destination "/root/minecraft-checkpoints/$C" \
  --checkpoint-id "$C" \
  --source-allowlist reports/server_only/source_archive_allowlist.json \
  --source-reference reports/server_only/source_reference.json
```

source参照はRelease411682582・asset637059866、SHA256 `97b5d9f0862449c881cd5804485f1a27cc5164baf3f51df09c66c2658ac3d6da`。保存先は新しい未公開Draft Releaseだけとし、既存Release・tag・assetを上書きしない。新checkpointの再取得hashと新規ディレクトリへの復元・代表CPU検証を記録する。GitHub保存、実験の実行状態、旧ARR原本不足によるPARTIALは別々に報告する。
