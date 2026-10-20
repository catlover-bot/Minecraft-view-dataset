Goal015C：schemaキー順序の局所検証と12呼出し診断の未承認準備 — 要件監査

本監査はユーザー添付 `7dee746e-4b7b-4ffc-80de-eb7dd3668846/pasted-text-1.txt` の全文と、現行コード・個別テスト・保存済み証拠を照合したもの。対象は準備の完成度であり、実モデルの成功や過去carve-onlyの原因を認定しない。**実モデル呼出し0、本番承認0、本番launcher実行0、実Draft操作0。** 実PTYテストの入力とfake transportの出力は隔離fixtureのみである。

監査対象plan IDは `95ef8e4c7e0d8e261087944374e1a690f32516caa91cf06fd95740a75b453503`、plan bytes SHA256は `065fe859b152912e41e0b5ef8697fa11c6e2764a862a7f23528654a8e7a21995`（25,074 bytes）。canonical rootは `/root/minecraft-runs/goal015c-20261020T094646Z-fcb2ec03`。実装67ファイル、wire6種類、12予定行、prompt・task・spec・旧状態へのbindingを[EXECUTION_BINDING.json](EXECUTION_BINDING.json)で公開する。read-onlyの `planning.verify_plan`、`pricing.verify_evidence`、`runtime.status` はこの実物に対して一致を確認した。planの `api_allowed=false`、`budget_approved=false`、`fixture_only=false` と、approval/authority/ledger/実行lock未作成を確認した。

以下で「確認済み」は局所CPU・保存証拠・コード読取り・fixtureの範囲を表す。「将来」は利用者の別承認後だけに許される動作。初回監査後に実装commit/pushと本番read-only preflightを完了し、下表へ追記した。最終公開状態は[README.md](README.md)から辿る。

**証拠とテストの照合**

[FINAL_TESTS.junit.xml](FINAL_TESTS.junit.xml)を実際に解析し、129 testcases、failure 0、error 0、skip 0を確認した。[FINAL_TEST_RECEIPT.json](FINAL_TEST_RECEIPT.json)の記録する全17コード・テストhashは監査時の実物と一致する。単に件数を合格根拠とせず、下表の各要件をテスト名と検査内容に対応づけた。

| テストファイル | 実行case数（parametrize展開後） | 主な確認対象 |
| --- | ---: | --- |
| [test_local_order.py](../../../offline_evaluation/experiments/goal015c/tests/test_local_order.py) | 28 | 旧Request.data境界、A/B順序差、C許可差分、再ソート拒否、wire直接送信 |
| [test_evaluate.py](../../../offline_evaluation/experiments/goal015c/tests/test_evaluate.py) | 28 | 3操作8セル、失敗分類、strict評価、既知usageからの費用照合 |
| [test_runtime.py](../../../offline_evaluation/experiments/goal015c/tests/test_runtime.py) | 19 | 12scope、予約、unknown、再開、copy/authority、実PTY、raw耐久保存 |
| [test_pricing.py](../../../offline_evaluation/experiments/goal015c/tests/test_pricing.py) | 17 | 24h、各文書時系列、同契約更新、本文hash、レビューと公式URL |
| [test_launcher.py](../../../offline_evaluation/experiments/goal015c/tests/test_launcher.py) | 16 | hidden入力、2回preflight、1回run、EOF/誤承認、終了status |
| [test_checkpoint.py](../../../offline_evaluation/experiments/goal015c/tests/test_checkpoint.py) | 14 | archive/restore、Draft/asset再利用、改変・path traversal拒否 |
| [test_cli.py](../../../offline_evaluation/experiments/goal015c/tests/test_cli.py) | 7 | finish/save-only分離、保存失敗回復、partial非0、completion lock |

最終129件は新Goal015C関連fixtureだけを、既存 `.venv-server-v2-20261013`、Python `-B`、thread1、pytest cache無効で実行した。CPU 17.119489秒、wall 19.658126秒、最大RSS 54,984 KiB。[conftest.py](../../../offline_evaluation/experiments/goal015c/tests/conftest.py)は全fixtureでsocket connect/connect_ex/create_connection/getaddrinfoを拒否する。旧全suite、数万ファイルの環境再監査、installは実施対象ではない。

追加の[BOUND_DEPENDENCIES_FIXTURE.json](BOUND_DEPENDENCIES_FIXTURE.json)は、実planの67実装依存、実取得11文書、task/prompt/spec/evidenceを含む別fixture planを使った確認である。mock12件・CPU task success12行→評価→専用archive→fake Draft/asset→再取得失敗→同checkpointを再利用した再取得・復元までPASS。fake Draft1、asset1、保存によるledger bytes変更なし。CPU 2.793459秒、wall 2.918645秒、最大RSS 49,632 KiB。実planのcontrolsは未作成のままである。この12件を実研究APIの成功件数に加算しない。

**Phase 1 — どの境界でキー順序が変わるか**

根拠は[LOCAL_ORDER_AUDIT.json](LOCAL_ORDER_AUDIT.json)、[EVALUATOR_ORDER_READINESS.json](EVALUATOR_ORDER_READINESS.json)、[local_order_audit.py](../../../offline_evaluation/experiments/goal015c/local_order_audit.py)の `run_audit/schema_orders/capture_old_fixture`。

| 要件 | 現物で確認した内容 | 直接対応する検査 |
| --- | --- | --- |
| 元schema→保存payload→clone→最終bodyをorder-awareに比較 | 元schemaとpure builderはop-first。選択したGoal015保存payloadは既にsorted。`json.loads(canonical_json(payload))` はpure buildからもpropertiesをsortし、最終Request.dataでもsortedになる | `test_actual_old_request_data_sorts_properties_but_not_arrays`。保存物と再構成の各stageを別記録 |
| set/fill/carveそれぞれの差 | set `op,x,y,z,block`→`block,op,x,y,z`。fill `op,x1,y1,z1,x2,y2,z2,block`→`block,op,x1,x2,y1,y2,z1,z2`。carve `op,x1,y1,z1,x2,y2,z2`→`op,x1,x2,y1,y2,z1,z2` | `schema_orders` はoperation別properties列を保存 |
| required/anyOfをpropertiesと混同しない | 全stageでrequired配列とanyOf配列の順序は不変。typed schema treeは等価でもordered representation hashは異なる | 同テスト、および `test_A_B_only_properties_order_and_C_only_format_removal` |
| 旧O比較は両Builder各1件だけ | Goal015Bで既選択の2件と既存object hashに一致。古いnative builderとserializerから再構成 | `historical_wire_evidence=RECONSTRUCTED_FROM_FROZEN_CODE`、`historical_original_wire_bytes_available=false`。親72rawの再診断0 |
| actual Request.dataをisolated transportで捕捉 | 旧経路はダミー環境mappingとfake transportのみで実Request生成まで通す。新経路は全6 Builder/conditionで固定wireと一致 | `test_old_capture_requires_explicit_fixture_scope`、`test_new_sender_uses_exact_wire_and_saves_proof_before_transport` |
| Bを直前に再ソートしたら拒否 | 型・値の等価だけでは送信許可しない。expected body bytesとの完全一致に失敗し、資格情報取得より前に停止 | `test_resorting_B_before_send_is_rejected_before_credentials` |
| 出力raw順序・echoを配送証明にしない | `output_raw_key_order_used_as_schema_evidence=false`、`remote_delivery_attested=false` | reportの明示フィールド。旧wireの完全bytesも独立remote受信証明もない |

確定したのはローカルserializerのコード挙動である。property orderingが旧48件のcarve-onlyを生じさせたか、provider内部で何が起きたかは未確定のまま。旧評価の訂正、旧72件の置換、モデル変更の根拠にはしていない。

**Phase 2 — 1つの固定ミニ課題と独立CPU正例**

| 要件 | 実装・証拠 | 直接対応する検査 |
| --- | --- | --- |
| 添付の英語summary、JSON入力、凍結system promptをそのまま使用 | [EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md)、[adapter.py](../../../offline_evaluation/experiments/goal015c/adapter.py)の `TASK_SUMMARY/input_text/build_payload`、planのtask/prompt descriptor。実建築、旧S/F、画像を使わない | A/B/C payload比較、`planning.verify_plan` のexact task binding |
| fill→carve→setの3操作、stone7+glass1 | [TASK_CPU_PROOF.json](TASK_CPU_PROOF.json)は8つの実座標・材質と独立traceを保存。旧executorと独立疎辞書executorが一致 | `test_three_operations_eight_exact_cells`（両Builder・3条件） |
| primaryはgrid完全一致、操作要求は別項目 | [evaluate.py](../../../offline_evaluation/experiments/goal015c/evaluate.py)の `grid_exact/task_pass` と `operation_sequence_conforms` を分離 | `test_exact_grid_is_separate_from_operation_sequence` |
| 空白・object順序を採点しない。座標・材質を有利に修復しない | 元world座標の直接比較。literal材質と旧validator正規化値が違えばtask successにしない。alignment/rotation/GT修復なし | `test_whitespace_and_object_key_order_do_not_change_answer`、`test_material_alias_not_repaired_into_task_success` |
| 課題成功を研究Nや機構証明にしない | 仕様で合成課題1種類、2予定反復と固定 | 仕様の解釈制限。実建築canaryへの移行経路なし |

**Phase 3 — 3条件・2 Builder・2反復、固定wireとscope**

| 要件 | 実装・証拠 | 直接対応する検査 |
| --- | --- | --- |
| AはGoal015送信順、Bは3つのproperties辞書だけ元順へ戻す | `adapter.build_payload/verify_conditions/order_diff`。A/Bのrequired、anyOf、enum、型、name、strict等は同じ | `test_A_B_only_properties_order_and_C_only_format_removal` |
| A/Bで受理集合不変、生成分布不変とは主張しない | schemaの辞書列挙順以外を変更しない検査と、同じ正例・負例を両schemaへ適用 | `test_schema_acceptance_unchanged_by_property_order`。仕様が分布の同一性を仮定しない |
| Cはnative format指定だけを削除 | OpenAI `text.format`、Anthropic `output_config.format` と空の親だけ削除。JSON mode/tools/例示を追加しない | `verify_conditions` のC byte比較、同A/B/C差分テスト |
| 両snapshot/settings/cap/timeout/endpoint/tierを固定 | `pricing.contract`、`adapter.validate_wire`、6種類の[固定wire](wires/)。O `gpt-5-mini-2025-08-07` medium、A `claude-haiku-4-5-20251001` temperature1/thinking disabled、cap32768、timeout600 | `planning.verify_plan` が完全contractとbodyを検査。wireテスト6条件。capやモデル変更は同planに通らない |
| 最終bodyをcommit前に固定し、そのbytesを送る | `Request(endpoint,data=wire,method='POST')`。canonical object hash、ordered representation hash、body SHAを別に保存 | `test_new_sender_uses_exact_wire_and_saves_proof_before_transport`、`test_durable_proof_failure_prevents_dispatch` |
| 診断IDを親callに混ぜない | `goal015c_call_001`〜`012`。固定順はrepeat1 O/A,A/C,O/B,A/A,O/C,A/B、repeat2 A/B,O/C,A/A,O/B,A/C,O/A | [planning.py](../../../offline_evaluation/experiments/goal015c/planning.py)の `EXPECTED_SCHEDULE/verify_plan`、`test_wire_tamper_copy_scope_and_authority` |
| concurrency1/retry0、予定反復とretryを区別 | 逐次loop、各NOT_SENT IDだけ1回dispatch。transportにretry/redirect経路なし | `test_full12_durable_wire_raw_and_duplicate`、`test_unknown_holds_and_resume_only_not_sent` |
| 出力失敗なら残り条件を固定して続行、運用エラーは停止 | runtimeのknown output failure分類とSAFE_STOP、contract failureの再開拒否。認証/HTTP等は同じtransport例外停止経路 | `test_known_bad_output_continues_all_conditions`、`test_known_model_failure_continues_but_unknown_contract_does_not`、通信unknown fixture。実認証/HTTP障害の照会はしていない |

**Phase 4 — 未承認予算・契約・将来起動の安全要件**

| 要件／安全条件 | 現在の実装・保存証拠 | 直接対応する検査 |
| --- | --- | --- |
| 公開公式文書GETと明示review/sealだけ | [PRICING_EVIDENCE.json](PRICING_EVIDENCE.json)は11実取得本文のURL/実UTC/bytes/SHAとcapture/reviewを参照。[PRICING_REVIEW.json](PRICING_REVIEW.json)がsnapshot、native/text、sampling、cap、通常単価、未確認事項の読取り根拠 | [pricing.py](../../../offline_evaluation/experiments/goal015c/pricing.py) `capture/seal/verify_documents`。`test_capture_without_explicit_read_review_cannot_seal` |
| Cにnative=true契約を流用しない | 新contractのconditionsでA/B native=true、C native=false/text output/JSON_mode=false | exact contract equalityと各conditionのwire検査 |
| 実取得から24h、各文書hash・時系列・公式HTTPSを確認 | 最初のGET開始から24h。今回期限は **2026-10-21T10:00:58.172322+00:00**。公表snapshot終了日にも停止 | `test_current_and_boundary_expiry`、`test_future_date_rejected`、`test_real_body_tamper_rejected`、`test_missing_document_or_nonofficial_url_rejected`、`test_redirect_official_allowlist`、`test_seal_and_verify_check_every_document` |
| 同契約の明示更新でplan/予算をresetしない | launcher/CLIの `--pricing-evidence`→runtime `_pricing`。base planとauthority/ledgerは同じ。単価・contract変更は拒否 | `test_same_contract_explicit_renewal_does_not_change_plan`、`test_changed_price_requires_new_design`、`test_contract_hash_and_claimed_expiry_bound_to_actual_capture` |
| 新規USD3.25と旧費用を分離 | O 6×0.165536 + A 6×0.363840 = **3.176256**。親既知費用4.43765700を別表示、旧枠credit 0。400000/200000入力予約と32768出力capを縮めない | `test_reservation_is_full_context_and_output_cap`、`test_whole_call_reservation_and_separate_budget`。現物planも同額 |
| planはdisabled、実承認後だけ新controls作成 | [UNAPPROVED_STATUS.json](UNAPPROVED_STATUS.json)。新approval/authority/ledgerは未作成。plan flagsは将来もfalseでauthorityを別保存 | runtime `execute/verify_enabled`。`test_preview_is_read_only_and_full_12`、`test_disabled_finish_does_not_create_authority_or_completion` |
| canonical path/plan ID/code/spec/wire/旧状態をbindしcopy・改変を拒否 | public registryとplan bytes、67実装hash、wire descriptors、metadata binding、親plan/approval/authority/ledger/lock・未承認子planを照合 | `test_wire_tamper_copy_scope_and_authority`、`test_parent_lock_and_parent_change_during_approval` |
| 親・子との競合とfinish中resumeを防ぐ | 親existing lockをrbで先に取得。子enabled/execution/completionの不在を検査。新execution lockは承認後だけ。completion lockと生成を相互排除 | runtime `locks`、CLI `completion_lock`、`test_parent_lock_and_parent_change_during_approval`、`test_completion_lock_excludes_resume` |
| キー入力前preflight、missing keyのみhidden入力、再preflight後に1回だけrun | [start-reviewed.sh](../../../offline_evaluation/experiments/goal015c/start-reviewed.sh)のcanonical設置版はplan hash対象。資格情報を表示・log保存しない。空/EOFは停止 | `test_hidden_missing_keys_two_preflights_and_one_run`、`test_preflight_rejection_prevents_run`、`test_empty_or_eof_key_cancels`、`test_copied_launcher_refuses_before_prompts` |
| 実TTY、read/write別handle、flush、ID/金額は単一関数 | runtime `approval_text/confirm_tty`。この監査表は手入力する承認文を転載しない | `test_real_pty_rejects_enter_wrong_and_eof`、`test_real_pty_exact_approval_and_duplicate_execution`、`test_non_tty_read_failure`、launcher `test_no_controlling_tty_refuses` |
| 本番hook拒否、fixtureから実adapterへ行かない | 本番はadapter/confirmation/checkpoint hook不可。fixtureは明示fake adapterと `fixture_only=True/uses_live_transport=False` が必要 | `test_production_hooks_and_fixture_missing_adapter_refused` は実adapter moduleをfixtureへ渡しても送信前に拒否 |
| 予約→送信状態→Request.data証拠→HTTP→raw→精算をdurableに保存 | `_save/_raw_save` とdirectory fsync。before_send callbackがbody SHA/bytes/equality/公式hostを保存。raw保存後だけusage精算 | `test_raw_durable_before_settlement`、`test_full12_durable_wire_raw_and_duplicate`、`test_durable_proof_failure_prevents_dispatch` |
| unknown holdを残す、同IDを再送しない、authority確認後NOT_SENTだけ再開 | uncertain行は予約全額を保持しスキップ。NOT_SENTにorphan rawがあれば拒否。known costとhold合計でcap確認 | `test_unknown_holds_and_resume_only_not_sent`、`test_orphan_response_not_sent_refuses_resume`、重複実行・予算テスト |
| 各送信直前に契約・旧状態・期限再検査 | loop内で `verify_plan` と `_pricing`。承認待ち中に親状態が変わった場合もcontrols作成前に停止 | `test_pricing_expiry_before_approval_and_each_send`、親状態変更fixture |
| HTTP/response/request IDは存在する情報のみ、headers/キー非保存 | adapterのsafe metadataとraw bodyだけを保存。missing IDを生成しない。進捗にはcondition/state/known/holdだけ | `adapter._dispatch/send_wire` と `runtime._dispatch/_progress` のコード読取り。実provider識別子の取得試験は未実施 |
| 終了status JSON、partial/save failure非0、exit0をtask成功と同一視しない | CLIが最終JSONを出し、wrapperはCLI/tee両statusと一意log pathを表示。`settlement_does_not_imply_task_pass=true` | `test_run_exactly_once_and_partial_nonzero`、`test_run_save_failure_nonzero_and_explicit_upload_gate`、`test_save_failure_or_partial_propagates_without_retry`、`test_tee_failure_overrides_success` |

**Phase 5 — 評価と未公開Draft保存の将来経路**

| 要件 | 実装・証拠 | 直接対応する検査 |
| --- | --- | --- |
| 独立raw抽出→strict validation→旧/独立executor | `evaluate_one` は保存rawから独立extractしfrozen parserとも照合。raw/text descriptor、response identity、known usageと費用を再照合 | `test_saved_text_mismatch_cannot_be_success`、`test_known_cost_reconciles_against_raw_usage_and_rejects_changed_ledger_cost` |
| HTTP/finish/usage、JSON、operation種別件数、3操作適合、voxel数、grid完全一致、truncation/touch列 | `evaluate_one` の初期rowと計算列。gridと操作順を別判定し、unknown列をNULLに維持 | 3操作正例、`test_provider_completion_is_separate_from_protocol_validity`、`test_exact_grid_is_separate_from_operation_sequence` |
| empty/carve-onlyはprotocol-validでもtask不合格 | 実行後の元座標8セル完全一致が必要 | `test_empty_and_carve_only_valid_but_not_task_success` |
| 未完JSON・fence・型を修復しない。truncationを実行しない | strict parser。parseableなtruncated programの操作件数は診断列だけで、最終gridはNULL | `test_truncated_json_stays_failure_with_null_grid`、`test_parseable_but_truncated_program_is_never_executed`、`test_no_fence_removal_json_repair_or_type_coercion` |
| touch超過を列挙前に拒否、CPU失敗をモデル失敗にしない | validate_actions、独立executor照合、別status | `test_touch_limit_rejection_without_execution`、`test_cpu_error_not_known_model_output_failure` |
| known failure/UNKNOWN/NOT_SENT/CPU不能/operationalを区別 | 未送信・unknown・CPU不能のtask結果はNULL。モデル不一致はoperational | `test_unsent_uncertain_not_scored_as_failure`、`test_model_mismatch_operational_not_task_failure` |
| 全12行、各conditionの実送信・可能送信・完了・settlement・成功・費用 | `summarize` が予定分母を保持し `post_returned/possibly_sent/completed_response/protocol_valid/strict_success/known_cost_rows` を分離 | `test_all_planned_conditions_preserved_in_descriptive_summary`、`test_explicit_finish_partial_preserves_null_remaining` |
| 事前解釈表、統計的優位性や機構証明なし | [EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md)に4結果パターンを固定。A/B失敗C成功もprovider側promptを含む差として限定 | evaluatorは記述集計のみ。bootstrap/p値/ランキング・親残8件/旧80件への自動移行なし |
| 専用archive、再現依存、controls/raw/evaluationをbindして安全に復元 | [checkpoint.py](../../../offline_evaluation/experiments/goal015c/checkpoint.py) `create_checkpoint/verify_archive`。Python環境等の外部依存を明記し、旧provenance全treeは再帰収集しない | `test_rehashed_archive_controls_rejected`、`test_traversal_and_links_refused_before_restore`、`test_repeated_save_only_does_not_expand_checkpoint_history` |
| 新Draftは明示allow-create、既存release/assetを勝手に上書きしない | corrected frozen uploaderを再利用。list→release ID→asset IDでdownloadしSHA/restoreを確認 | `test_missing_draft_requires_explicit_creation`、`test_existing_release_identity_never_overwritten`、`test_mock12_evaluate_archive_upload_failure_then_save_only_retry` |
| 保存失敗から生成せず同checkpointで回復 | [cli.py](../../../offline_evaluation/experiments/goal015c/cli.py) `finish/save_only`。completion progressにcheckpointとproofを保存、再実行はsaveだけ | `test_finish_retry_reuses_same_checkpoint_and_no_generation`、`test_save_only_reuses_failed_finish_and_no_evaluation`、BOUND_DEPENDENCIES_FIXTUREの同asset再利用 |
| partial/unknownも保存可能だが完了と呼ばない | explicit finishは12行を保存しPARTIAL_SAVED・非0。runは12SETTLED/unknown0の場合だけ自動finish | `test_unknown_transport_and_lower_fixture_budget_remain_saveable`、`test_explicit_finish_partial_preserves_null_remaining` |

**保全・公開・完了報告の境界**

| 要件 | 監査時点の状態／証拠 | 最終確認の扱い |
| --- | --- | --- |
| 旧コード・親子plan/ledger/approval/authorityを変更しない | ENTRYの70paths、scoring/local-order補足とのunionは76paths。親現物は72SETTLED・8NOT_SENT・unknown0・USD4.43765700、子はdisabled | [PRESERVATION.json](PRESERVATION.json) の2026-10-20T10:29:24.051292+00:00閉鎖receiptを確認。76pathsの全before/after一致、67実装・129test・6public wire一致、必要controls不在を記録。対象外の旧全treeまで確認したとは主張しない |
| 新領域だけで作業、未commit作業を保全 | 新コード・新report・専用run。旧残8件を開始しない。reset/clean/recloneや既存artifact削除を要しない | 最終diff/clean判定はREADMEとprivate `PUBLICATION_RECEIPT.json` |
| 研究API、model-list/auth/token-count、キー照会、実承認、既存Draft変更なし | local audit、129 fixtures、BOUND fixture、UNAPPROVED_STATUSが各scopeで0を記録。公開文書GETは料金資料取得として別扱い | model account availabilityはNOT_RUN。現在の文書は過去provider内部仕様を証明しない |
| 通常commit/push、専用branch、remote一致 | 実装commit `7f4ab6f4b13627d6a6009894635640f60a6e08a6` を通常pushしremote一致を確認 | [POSTCOMMIT_PREFLIGHT.json](POSTCOMMIT_PREFLIGHT.json) はcommitted registryを使った本番 `preflight --require-unstarted` のPASS。76原本も再照合し、controls不在。launcher/start/runは実行していない |
| 最終SHA、remote/clean、実在launcherの1行、期限 | READMEが公開結果と利用者向け起動コマンドを提供する。承認文は `approval_text` の実表示に従い推測転記しない | 最終push/remote照合はprivate `PUBLICATION_RECEIPT.json`、public manifest/copy照合は `MANIFEST.json/PUBLIC_COPY_RECEIPT.json`。作成前に成功したとは記載しない |
| 完了冒頭と7必須事項 | 冒頭文、順序差と証明限界、未確定原因、3条件/3操作8セル/固定12順、未承認独立予算、tests/code-plan-wire、実在起動行/期限、API0/承認0/保全/残8保留 | READMEと最終ユーザー報告で満たす。実モデル結果や旧原因の確定を追加しない |

要件別にコード・fixture・凍結planを照合し、必要76ファイルの保全、実装commit/push、commit後read-only preflightを確認した。本追記とmanifestを含む最終commitのremote/cleanはcanonical `PUBLICATION_RECEIPT.json` に保存する。これらは生成承認の代わりにはならず、実研究APIを実行して準備を「確認」することもない。
