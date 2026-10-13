# Goal003 — 保存済みsmoke応答の集計修正と診断

初回smokeの集計例外を最小修正し、保存済み原本だけで全16計画入力・全14試行を再集計した。研究APIの追加送信は0回。実験は **PARTIAL** で、本評価960入力は未承認・NOT_RUN。保存の完全性と実験完了は別に扱う。

作業はroot接続、`research/server-only-20261013`、開始HEAD `26aadf04f45e3d529d094385b4538730d8725fb1` を継承した。実効設定は `workspace-write / auto_review`。通常コマンドの成功を確認したが、OSレベルの隔離を独立に保証する検証ではない。実行先は `/root/minecraft-runs/goal003-smoke-diagnosis-20261013T165339Z`。[実行条件](execution_context.json)、[事前照合](preflight.json)、[診断ファイルSHAとコードSHA](diagnostic_run.json)を保存した。

例外の実際の不一致は `completed_request_outcome`。OpenAI/building060/O、input `87b41311abf64f23c9c8e5df50e5e804847890062f80d4e3560762a71332cf25`、attempt `416fa64024984552b76b887c1c4f6a55` で、旧collectorが算出したexpectedは `false`、保存済みevaluationのactualは `true` だった。requested modelは `gpt-5-mini`、raw/stateのresponse modelは `gpt-5-mini-2025-08-07`。凍結済みpricingはこの2 IDを明示的に承認しており、評価側は既存helperで受理する一方、集計側だけが完全一致を要求していた。OpenAIの全8最終行でこのfieldのみ不一致、Anthropicは不一致0だった。[例外再現記録](exception_reproduction.json)。

修正は `analysis_v4.collect_planned_outcomes` の判定を `providers.accepted_response_model_ids(config.provider, config.model, config.pricing)` に統一しただけである。[最小diff](aggregation_fix.diff)。新しいsnapshot許可、部分一致、evaluationの無検査採用、原本書換えはない。生成時に固定した11ファイルの実装hash、input/attempt/provider/provenance/metricの検査を維持した。

| Provider | 計画入力 | 試行 | Retry | 成功入力 | 最終失敗 | NOT_RUN | 運用上測定不能 |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenAI | 8 | 11 | 3 | 6 | 2 | 0 | 0 |
| Anthropic | 8 | 3 | 1 | 0 | 2 | 6 | 0 |
| 合計 | 16 | 14 | 4 | 6 | 4 | 6 | 0 |

14試行は10入力への試行であり、14個の独立入力ではない。成功はactionのparse・validate・execute成功を意味し、再構築品質の高さを意味しない。P条件の2成功は `operations: []`、voxel数0の空グリッドだった。

| 建築 | Provider | O | J | P | E |
|---|---|---|---|---|---|
| 060 | OpenAI | 成功 | 成功 | 空グリッド成功 | 切詰め失敗 |
| 075 | OpenAI | 成功 | retry後成功 | 空グリッド成功 | 切詰め失敗 |
| 060 | Anthropic | parse失敗 | parse失敗 | NOT_RUN | NOT_RUN |
| 075 | Anthropic | NOT_RUN | NOT_RUN | NOT_RUN | NOT_RUN |

各input ID・最終attempt ID・指標は[全16行JSON](planned_outcomes.json)と[CSV](planned_outcomes.csv)。各応答のモデルID・response ID・raw SHA・長さ・usage・終了理由・再parse結果は[全14試行JSON](attempts.json)と[CSV](attempts.csv)に記録した。

OpenAIのE条件2入力は、それぞれ初回とretryの計4応答が `status=incomplete`、`incomplete_reason=max_output_tokens` だった。各output token数は16,384、記録されたreasoning token数は順に832・640・576・1,088。切詰めはrawの終了理由で確認したもので、token数だけからreasoning枯渇を推測していない。`non_reasoning_output_tokens` は記録値の差分であり、可視textを独立に再tokenizeした数ではない。

OpenAI/building075/Jの初回はstrict JSONを通るが、operation index3の`carve`に余分な`block`があり、`operation 3 fields do not match carve schema` でvalidation失敗した。2回目は成功した。その他のO/J成功出力はrawからparsed・voxelまでCPUで再実行して保存物と照合した。全14応答に識別済みresponse・既知usage・抽出可能なtextがあり、観測された失敗は返却textの形式/schema違反と明示的切詰めだった。transport失敗・text抽出失敗・refusal blockは観測されなかった。

Anthropicの3応答はすべて `stop_reason=end_turn` で自然終了していたが、JSON全体をMarkdownフェンスで囲んでおり、主評価のstrict JSONに失敗した。building060/Oの初回とretryは、一意な外側フェンスだけを除く感度分析で32/37操作、1,411/2,334 voxelsまで実行可能だった。building060/Jにはさらに2つのobjectで重複`z1`キーがあり、フェンス除去後もstrict JSONに失敗する。感度分析は主評価のstatusや指標を変更していない。JSON補完・括弧修復・値推測・GTによる補修は行っていない。

`completed_request_outcome` はID・承認モデル・既知usage・保存rawを持つ識別済み返却応答の判定であり、providerの自然終了とは異なる。切詰め4試行は前者を満たすが `natural_response_complete=false`。既存のfailure0方針に従い、最終的な切詰め/parse失敗だけをbounded IoUの0に含める。NOT_RUNは0にせず除外し、失敗のMAE・repairはnullのままとする。

| OpenAI条件 | 有効入力/独立建築 | aligned IoU | dimension MAE | material set F1 | aligned正規化repair |
|---|---:|---:|---:|---:|---:|
| O | 2/2 | 0.2463644509 | 10.5 | 0.7333333333 | 1.2977610771 |
| J | 2/2 | 0.2468572090 | 10.0 | 0.7333333333 | 1.4632382543 |
| P | 2/2 | 0 | 32.5 | 0 | 1 |
| E | 0/0 | null | null | null | null |

上表は有効出力のみの建築等重み平均。成功6件すべてでmetric artifact・run/attempt identity・GT hashを検証し、strict translationの全metric辞書をCPU再計算して完全一致した。[検証記録](metric_verification.json)。Eのfailure0 IoUは分母2で0。Anthropic O/Jは有効出力0、failure0 IoUは各分母1で0、各条件の残り1入力はNOT_RUN。Anthropic P/Eは各2入力ともNOT_RUNで、分母0・指標null。provider間common-validは全条件で0。smokeではp値・信頼区間・モデル優劣の一般化を付けていない。[全集計](summary.json)。

停止位置はAnthropic/building060/Jのattempt2予約前。共有retry上限4回を使い切り、`SharedBudget.reserve` が `shared_retry_limit` を返し、generationがproviderループ全体からreturnした。そのため後続の未実行6入力へ到達しなかった。fixtureで同じ11＋3試行の形、後続6件のNOT_RUN、null haltでの非完了を確認した。実行順やretry方針は今回変更していない。[停止診断](stop_diagnosis.json)。`halt_reason=null` とgeneration exit0は全16入力完了を保証しない。

凍結済み `smoke.dry_run_smoke_campaign` はaccount availabilityを常にNOT_RUNと表示する。診断ではrawに基づきOpenAI11、Anthropic3の承認モデル識別済み応答と、action成功6/0を履歴として記録した。現在・将来のavailability照会や保証は行っていない。[履歴](availability_history.json)。

| Provider | Input tokens | Output tokens | 台帳上の保守的USD |
|---|---:|---:|---:|
| OpenAI | 58,089 | 105,452 | 0.22542625 |
| Anthropic | 4,173 | 5,795 | 0.033148 |
| 合計 | 62,262 | 111,247 | 0.25857425 |

合計173,509 tokens、calls14、retry4、unknown usage0。raw・local state・共有ledgerのusage/費用/identityを既存validatorで照合した。上表は請求確定額ではない。[台帳照合](ledger_reconciliation.json)。原本69証拠ファイルと4制御ファイルを区別し、[修正前SHA](originals.before.json)と[診断前後SHA](original_hashes.json)を照合した。制御ロック以外に原本への書込みはない。

全server_v2スイートは **307 passed、failure/error/skip 0**。既存242件＋新規65件で、実モデル完全一致・承認/未承認snapshot・retry/最終失敗/NOT_RUN混在・実際のidentity/hash改変・retry上限の早期return・診断transport禁止・forensic保存境界を含む。[結果](test_results.json)、[ログ](tests.log)、[JUnit](tests.junit.xml)。初回の対象テストはpytest既定の`/tmp`がアプリの既存許可root外だったため失敗し、コードやskipを変更せずrepository内basetempを指定して成功した。その履歴も結果JSONに残した。旧6 collection errorsは[元ログとSHA参照](legacy_suite_reference.json)およびコピーを保存し、一括skipやダミー実装で消していない。

修正・診断コードと報告をcommit `044ae34349892caa2d6399c394a2d7464efa8961` として専用ブランチへpushした。[新Draft Release 411921452](https://github.com/catlover-bot/Minecraft-view-dataset/releases/tag/untagged-3cf52059b9f2d7f1b715)に原本のfreeze/identityを変更しない **読取り専用forensic snapshot** を保存した。archiveは147ファイル、SHA256は `8f60b4df66f11cec77093d04c8f077652bc65cf42219ffbcf2161d117481ec8e`。既存大容量source assetはrelease `411682582` / asset `637059866` / SHA `97b5d9f0862449c881cd5804485f1a27cc5164baf3f51df09c66c2658ac3d6da` の参照を再利用し、現在のGitHub digestも一致した。

archiveと6 sidecarをサーバーの別ディレクトリへ再取得し、全assetのSHAと復元した147ファイルすべてを照合した。原本69ファイルも一致し、復元したcampaignは `campaign_ledger_directory_mismatch` で受理されないことを確認した。archive完全性COMPLETEと実験PARTIALは別であり、復元物は生成再開用の複製ではない。[再取得検証](github_verification.json)は新しい8番目のassetとして保存・再取得照合済み。[保存索引](checkpoint_index.json)、[最終の原本・台帳照合](final_validation.json)に確定値を記録した。archive自身のhashを検証する後続記録は、自己参照を避けてsidecarと後続報告commitに保存する。

次は、JSON出力形式とE条件の出力量を見直し、P条件の空出力の原因を点検したうえで、**変更したprotocolの再smokeを明示承認する提案**が必要と判断する。未実行6件の初回送信だけでは、既に観測した形式違反・切詰め・空出力の問題を解消できない。また現在のループは未実行6件の前でretry拒否によりreturnする。旧campaignはPARTIALの原本として保持し、変更案では旧ledgerの累積calls14・retry4・173,509 tokens・USD上限0.25857425を引き継いで、新規calls/retryの許容量を別途明示する。残額や残り6 callsを新retry枠の承認と扱わない。今回、有料生成の開始・再試行・モデル照会は実行していない。

CPU診断を再検証する場合は、`run_cpu_diagnosis.py --destination <既存診断ディレクトリ> --verify-existing` を用いる。これは研究transportを禁止してメモリ上で再計算し、保存済みJSONと照合する。新規診断を作る場合だけ未使用destinationを指定し、既存成果物を上書きしない。
