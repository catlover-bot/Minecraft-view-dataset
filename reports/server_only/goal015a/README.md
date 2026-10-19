Goal015Aの対象は、親Goal015の未送信073〜080だけである。親72件のCPU評価と新しい非公開Draft保存を完了し、**親と補完を合算してUSD7.00**を提案する別承認入口を用意した。子planは未承認・API無効で、子の実行用approval・authority・ledgerは未作成。この準備で研究API送信、キー照会、モデル照会、token-count APIは0。

親は `SAFE_STOP / next complete paired block exceeds approved cap`、72 SETTLED・8 NOT_SENT・unknown0、計上USD4.43765700・保留予約0のまま保全した。親plan IDは `5b80fdef342dd2b9f68b3c2bec2d3d83af2731c3b5a9c8940959846ea11c68ce`、SHA256は `6533090e07090b7279f0383937631c609b17c84c922f8cd8deadcd2b9ea22257`。親のUSD5承認・台帳・凍結コードを変更していない。

実際の保存応答評価は次のとおり。全80予定行を保持し、未送信8件を0点にしていない。正式評価の原本は `/root/minecraft-runs/goal015-20261017T170546Z/evaluations/goal015a-parent72-20261019T141842Z-13cfbc79/evaluation_80.json`、SHA256 `e430c61b0eaa1d6108158636fb4cbff09edf9cc481d6766c2445460ba3ce2420`。再生成なしの既存evaluate CLIによるCPU評価で、CPU10.05秒・経過10.06秒・最大RSS83,220KiB。

| 親80予定行の状態 | 件数 | 解釈 |
| --- | ---: | --- |
| protocol-valid・正常empty | 48 | 正常emptyをモデル出力失敗へ再分類しない |
| protocol-valid・非empty | 0 | 非空再建築成功は0 |
| モデル出力失敗 | 24 | truncation22、validation_failure2 |
| 未送信 | 8 | 指標NULL、計測対象外 |
| 計測失敗 | 0 | 指標計算は正常empty48件でPASS |

正常empty48件はoccupancy precisionが定義上NULL、他の5既定指標は定義済みである。両Builderとも36精算済みで、OpenAIは正常empty35・失敗1、Anthropicは正常empty13・失敗23。これは9建築までの途中記述で、全10建築完了の結果ではない。

utilityの有効ペアは各Builder18・寄与9建築。common-validはOpenAI17ペア・9建築、Anthropic2ペア・1建築で、後者のCIはNULL。定義されたIoU/surface F1のS/F差は全て0だが、正常出力が全てemptyという結果を条件の同等性・情報効果なしの証明と解釈しない。[詳細内訳](PARENT_EVALUATION_COUNTS.json)。

評価を含む新checkpointを既存CLIで作成し、新規非公開 **Draft417384305 / asset656494745** へ保存した。archiveは2,042,963 bytes、SHA256 `c16c6e954518808e4bc4bb56f777599099693beef65801df2779bca5e5e70a3c`。GitHub digestと再取得SHAが一致し、安全展開297ファイル、親raw/text144参照、評価・summary対応を検証した。旧指定archive（SHA `cecf6f23e1f3ac6309ed19d69c4ee8a4b94c0f6b653c1f5f69984707e9d0ffbb`）を保持し、既存17 Draftのmetadata/assetsも保存前後で不変だった。[保存証拠](PARENT_SAVE_AUDIT.json)。

archiveには元description、S/F入力・provider payload、GT voxel/bbox、凍結実装・Python環境、参照料金本文を一式同梱していない。これらはhashで識別する外部依存であり、archive単独での完全再現を主張しない。

補完scopeは全て `buildings_100_v4_building_047`。元schedule、payload実bytes、source/GT、model snapshot、schema/prompt、sampling/reasoning、output cap32768をそのまま保持する。

| 順序 | Builder | 反復 | 条件順 |
| --- | --- | ---: | --- |
| 073–074 | Anthropic | 1 | F→S |
| 075–076 | OpenAI | 1 | F→S |
| 077–078 | OpenAI | 2 | S→F |
| 079–080 | Anthropic | 2 | S→F |

| 固定料金・保守予約による計算 | USD |
| --- | ---: |
| 親の計上額 | 4.43765700 |
| 旧USD5までの残額 | 0.56234300 |
| 次Anthropicペア予約／旧上限への不足 | 0.727680 / 0.16533700 |
| 残8件全予約 | 2.117504 |
| 親＋残8件全予約 | 6.55516100 |
| 提案する親子合算上限 | 7.00 |
| 旧上限からの増分 | 2.00 |
| 子に使える最大残枠 | 2.56234300 |

追加USD7を使う計画ではない。予約は実費予測や完走・出力成功の保証ではなく、過去の約USD1.10という過小な外挿も再利用していない。旧991 historical calls／USD21.56083625は本予算へ加算しない。[scope・費用監査](SCOPE_COST_AUDIT.json)。

子canonical rootは `/root/minecraft-runs/goal015a-20261019T141842Z-13cfbc79`。識別子・plan SHAは [EXECUTION_BINDING.json](EXECUTION_BINDING.json)、唯一の関数から生成した承認文字列は [APPROVAL_TEXT.json](APPROVAL_TEXT.json)。実TTYで新承認を入力するまで送信できない。親子排他、親費用持越し、4隣接ペア予約、最大8追加送信／総80、retry0、concurrency1を固定する。unknownは予約を保持して停止し、明示resumeでもNOT_SENTだけを対象とする。

将来の完了時は親72行を認証して再利用し、子8行だけを評価する。元80 call_idへの一対一結合とSELECTED_SOURCESを別成果物へ保存し、親ledgerは変更しない。40予定ペア、2反復、最大独立N10、utility/common-validと各有効Nを旧集計器で保持する。新Draft保存・再取得検証まで成功した場合だけCOMPLETE_SAVEDとする。保存失敗時はfinishで保存だけを再試行する。

利用者がtmuxで実行する1行は次のとおり。この準備中は実行していない。

```bash
bash /root/minecraft-runs/goal015a-20261019T141842Z-13cfbc79/start-reviewed.sh
```

既定料金証拠は2026-10-19 **17:34:50.694342 UTC**までの24時間窓内でのみ利用可能。期限切れはキー入力前に拒否する。別の期限内・同一親契約のreviewed evidenceは `GOAL015A_PRICING_EVIDENCE` で明示できる。更新、状態確認、明示resume、保存だけの再試行は [OPERATIONS.md](OPERATIONS.md) を参照。

検証は新機能fixture・実PTYに限定し、旧全suiteは再実行していない。対象試験、起動script静的検証、原本281ファイルと55凍結コードの終了照合、未承認確認は [VALIDATION.json](VALIDATION.json) に記録する。コード・spec・起動scriptは子planのhash対象で、準備済みregistryのcommitも実行前に検証する。mainへ統合せず、既存Draftを公開・削除・上書きせず、この未承認状態で利用者の費用承認を待つ。
