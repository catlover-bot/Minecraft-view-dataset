Goal015Aは未承認の残8枠補完入口。親の旧start/resumeは使用しない。親planのUSD5、approval、authority、ledgerを書き換えて予算を増やすことはできない。

tmuxの実paneで、初回だけ次を実行する。

```bash
bash /root/minecraft-runs/goal015a-20261019T141842Z-13cfbc79/start-reviewed.sh
```

wrapperは親72/8/0・原本hash・commit済み子registry・料金鮮度・未実行状態を検証し、不足しているOPENAI_API_KEYとANTHROPIC_API_KEYだけを非表示入力する。キーをコード・ログ・ファイルへ保存しない。最後にruntime.approval_textの完全一致文を実TTYで要求する。空入力・不一致・EOF・非TTYでは送信しない。承認文は [APPROVAL_TEXT.json](APPROVAL_TEXT.json) を参照するが、パイプや自動入力で供給しない。

承認後は4つの元S/Fペアについて補完→CPU評価→新Draft保存・再取得検証を行う。ログはcanonical rootの新規 `start-reviewed-<UTC>-<random>.log` に保存し、終了時に実pathを表示する。partial、unknown、価格期限切れ、保存失敗は非0終了。COMPLETE_SAVEDと80精算・保存検証の結果を確認し、OS終了0だけを再建築成功と解釈しない。

読み取り専用の状態・承認previewは次で確認できる。いずれもキー・研究APIを使用しない。

```bash
cd /root/workspace/Minecraft-view-dataset
PYTHONDONTWRITEBYTECODE=1 .venv-server-v2-20261013/bin/python -B -m offline_evaluation.operations.goal015a.cli status --plan /root/minecraft-runs/goal015a-20261019T141842Z-13cfbc79/plan.json
PYTHONDONTWRITEBYTECODE=1 .venv-server-v2-20261013/bin/python -B -m offline_evaluation.operations.goal015a.cli preview --plan /root/minecraft-runs/goal015a-20261019T141842Z-13cfbc79/plan.json
```

既定の料金記録は `/root/minecraft-runs/goal015-20261017T170546Z/pricing-refresh-20261018T173449Z-fd5237eb/reviewed.json`。最古取得2026-10-18T17:34:50.694342Zから24時間の期限を実時計で確認する。起動前だけでなく送信前にも検査する。失効後に時刻やJSONを編集して延長してはいけない。

料金更新が必要な場合は、親planに対して既存CLIのcapture-pricingを新しい一意な出力directoryへ実行する。公式GETのみである。その本文を読み、料金、snapshot状態、API、structured output、reasoning/sampling、output上限が凍結契約と同じことを確認してから、確認済みcontract JSONと具体的review noteをseal-pricingへ渡す。captureだけは承認済み証拠ではない。変更がある場合は同じ契約としてsealせず停止する。

```text
python -B -m offline_evaluation.experiments.goal015.cli capture-pricing --plan <親plan.json> --output-dir <新しい価格取得directory>
python -B -m offline_evaluation.experiments.goal015.cli seal-pricing --plan <親plan.json> --capture <新capture.json> --reviewed-contract <本文確認済みcontract.json> --review-note <具体的な照合根拠> --output <新reviewed.json>
```

新reviewed.jsonの実pathを `GOAL015A_PRICING_EVIDENCE` で指定して初回wrapperへ渡せる。この値は料金証拠の指定であり、費用承認の代用ではない。旧証拠も保持する。

中断後は最初にstatusとローカルledgerを確認する。初回wrapperは既存実行状態を拒否する。明示的な再開が適切な場合だけ、既存子承認の同じscopeで以下を使う。`resume`はNOT_SENT以外を送らず、unknown予約を保持する。contract failureなど新承認が必要な停止を無理に解除せず、台帳の削除・初期化・手編集をしない。

```text
<既存Python> -B -m offline_evaluation.operations.goal015a.cli resume --plan /root/minecraft-runs/goal015a-20261019T141842Z-13cfbc79/plan.json --pricing-evidence <期限内reviewed.jsonの実path>
```

resumeは生成段階だけを扱う。8 SETTLED・NOT_SENT0・uncertain0となった後にfinishを実行する。生成が完了して保存だけ失敗した場合も、再生成せず同じfinishを使う。finishは既存の認証済み統合評価・同じcheckpointを再利用し、同一Draftのassetを上書きしない。同時finishは専用lockで拒否する。

```bash
cd /root/workspace/Minecraft-view-dataset
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv-server-v2-20261013/bin/python -B -m offline_evaluation.operations.goal015a.cli finish --plan /root/minecraft-runs/goal015a-20261019T141842Z-13cfbc79/plan.json --allow-create
```

partialやunknownの証拠を保存するだけならcheckpoint-localを使い、返された実directoryをcheckpoint-uploadへ渡せる。この操作は80完了の主張にならず、生成を再開しない。verification-dirは毎回新しいpathが必要で、失敗した検証証拠も保持する。既存Draftの探索・同一asset hash照合を省略するような手動uploadは行わない。

```text
<既存Python> -B -m offline_evaluation.operations.goal015a.cli checkpoint-local --plan /root/minecraft-runs/goal015a-20261019T141842Z-13cfbc79/plan.json
<既存Python> -B -m offline_evaluation.operations.goal015a.cli checkpoint-upload --checkpoint-dir <返された実directory> --verification-dir <新しい検証directory> --allow-create
```

親評価の新Draft417384305／asset656494745は今回の保全物で、補完用Draftと区別する。原description/provider payload、GT、凍結実装と環境、価格本文は外部依存。復元はarchiveの安全展開とhash照合であり、それだけで生成やgeometryを再実行した意味にはならない。
