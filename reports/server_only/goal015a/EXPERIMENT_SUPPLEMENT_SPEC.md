Goal015Aは、親Goal015の未送信8枠だけを補完するための未承認入口である。新しい科学実験、対象、条件、モデル、反復、指標は追加しない。この準備作業中の研究API送信・キー照会・モデル照会・token-count APIは0。

親canonical rootは `/root/minecraft-runs/goal015-20261017T170546Z`、親plan IDは `5b80fdef342dd2b9f68b3c2bec2d3d83af2731c3b5a9c8940959846ea11c68ce`、親plan SHA256は `6533090e07090b7279f0383937631c609b17c84c922f8cd8deadcd2b9ea22257`。親plan・USD5承認・authority・停止ledger・raw・凍結実装は変更しない。親への新しい評価・checkpoint・価格証跡の追記だけを許す。

対象は固定停止ledgerのNOT_SENT行 `goal015_call_073`〜`goal015_call_080`。全て `buildings_100_v4_building_047` の2 Builders×2条件×2反復である。元80行のcall_id、順序、paired_block_id、scene、condition、repeat_index、payload bytes、source/GT、prompt/schema、snapshot、sampling/reasoning、output cap32768をそのまま使う。精算済み72件の失敗を再送せず、3反復目を作らない。成績は対象や条件の選択に使わない。

隣接ペア順は、073–074 Anthropic repeat1 F/S、075–076 OpenAI repeat1 F/S、077–078 OpenAI repeat2 S/F、079–080 Anthropic repeat2 S/F。Sは原summaryだけのJSON、Fは親で固定した8人間可読フィールドのJSONで、管理metadataを除外した同じserializationを維持する。

凍結モデルは `gpt-5-mini-2025-08-07`（medium、temperature/top_p/seed省略）と `claude-haiku-4-5-20251001`（temperature1、thinking disabled）。旧adapter、parser/executor、評価関数、保存用の純粋helperを再利用する。親validatorのUSD5固定を変更・monkey patch・fixture化しない。

費用はDecimalで扱う。親計上額USD4.43765700、旧USD5までの残額0.56234300。次Anthropicペア予約0.727680に対する不足は0.16533700。残8件の最大予約は4×0.165536＋4×0.363840＝2.117504。親＋残8件の全予約は6.55516100。

提案上限は**親と子を合算してUSD7.00**。旧上限からの増分はUSD2.00であり、追加USD7ではない。子に使える最大残枠は2.56234300。これは凍結料金と保守的予約方式による上界で、実費予測・API完走・モデル出力成功の保証ではない。約USD1.10の旧参考外挿を残8件の見積もりへ流用しない。Goal007等の旧991 calls／USD21.56083625は本予算に混ぜない。

子planは正規pathとregistryに一意に束縛し、親plan・approval・authority・停止ledger・正式CPU評価・新Draft保存証明・checkpoint archive、実装/specとpayloadをhashで結ぶ。コピーから起動できず、親が固定状態から変われば停止する。api_allowed=false、budget_approved=falseのまま準備を完了し、実行用approval/authority/ledgerを作らない。

将来の初回送信は新しい実TTY承認が必要。承認文は新runtimeの唯一のapproval_text関数から生成し、preview・UI・試験・報告で共用する。親費用、合算上限7.00、追加上限枠2.00、残8件、retry0を示す。/dev/ttyのread/writeは別handleでflushし、空入力・不一致・EOF・非TTYは送信前に拒否する。Codexは実planへ承認を入力しない。

親の既存execution.lockと子lockを同時に保持し、親resumeと子生成の競合を防ぐ。送信前に少なくとも次S/Fペア全体を予約し、親計上額＋子既知費用＋未精算/unknown予約＋次ペア予約が7.00以内であることを確認する。子最大8送信、親と子の総送信最大80、retry0、concurrency1。未送信と証明できないcallは再送しない。未知usageは予約を保持して停止し、明示resumeでも正規子台帳のNOT_SENTだけを扱う。

価格記録は明示指定し、親契約との一致と最古取得時刻から24時間以内を送信前にも検証する。元のcapture-pricing／本文確認／seal-pricingで新証拠を追記できるが、時刻や上限を偽装しない。snapshot・料金・対応条件が変われば同じ契約として更新せず停止する。

今回の親CPU評価は途中状態の記述であり、全10建築完了の結果ではない。全80予定行を残し、未送信8はNULL。モデル出力失敗、正常empty、運用欠測、定義上NULL、計測失敗を分離する。正常emptyを不正出力へ再分類しない。

補完後は親72行と子8行を元80call_idへ一対一で結合し、selected sourceを別成果物で明示する。親ledgerのNOT_SENTを書き換えない。認証済み親評価を再利用し、子だけを旧評価定義で評価する。IoU/surface F1@1は同じoccupancy最適shift、utility/common-valid、scene等重み、paired2反復、planned10scene bootstrap2000/seed20261017/有限draw1900規約を継承する。最大独立N10、40予定ペアであり、best-ofや新独立対象8の扱いをしない。

親72件は評価を含む新checkpointを新Draftへ保存し、Release ID・asset ID・digest・再取得SHA・安全展開を検証する。将来の子と統合評価も別の新Draftへ保存する。保存失敗は保存だけを再試行し、生成を再実行しない。既存Draftを公開・削除・上書きしない。原description/input payload、GT、凍結コード等の外部依存は別に記録し、archive単独の完全再現を主張しない。

完了処理は専用のcompletion.lockで直列化する。保存進捗に同じ子ledger・統合評価・checkpointを束縛し、保存の再試行は同じcheckpointとDraft識別子を用いる。評価・checkpoint作成では親子lockも取得し、途中の変更を認めない。

利用者用tmuxスクリプトは不足キーだけ非表示入力し、新TTY承認後に補完→評価→保存を行う。partialや保存失敗では非0で終了し、OS終了0だけで全80完了と表示しない。この準備中はそのスクリプトを実行しない。CPU fixture・実PTYは隔離された偽plan/transportだけで実施し、mock呼出しを研究API送信に数えない。
