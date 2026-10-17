Goal015は、保存OpenAI説明の同じsummaryに他の7記述フィールドを追加する方針の、小規模paired development pilotである。今回の作業は準備だけで、研究API送信・モデル照会・キー照会は0、実生成とアカウント可用性はNOT_RUN。新USD5提案は未承認である。旧960 slotsや旧承認へ追加しない。

対象と順序は選択前commit `9c32add47cb724b1afc01924aa047cbd5619258a` のSELECTION_RULE.mdに固定した。元evaluation60の5 size_rank_stratumから各2件を選び、各層1件ずつ2巡に配置した。選択に過去IoU・CLIPScore・失敗・例示sceneを用いない。既知結果を持つ既存対象の再利用であり、盲検、preregistration、rest30の独立確認ではない。欠損時に別sceneへ置換しない。

| 提示順 | scene ID末尾 | 保存サイズ層 | 巡 |
| --- | --- | ---: | ---: |
| 1 | building_082 | 0 | 1 |
| 2 | building_003 | 1 | 1 |
| 3 | building_092 | 4 | 1 |
| 4 | building_025 | 3 | 1 |
| 5 | building_042 | 2 | 1 |
| 6 | building_052 | 2 | 2 |
| 7 | building_013 | 3 | 2 |
| 8 | building_024 | 4 | 2 |
| 9 | building_033 | 0 | 2 |
| 10 | building_047 | 1 | 2 |

全IDのprefixは `buildings_100_v4_`。保存OpenAI説明を固定する理由は全60件のsummaryが非空で、実験を小さく保つためである。有利な下流scoreによる説明元選択ではない。10建築×2 Builder×2条件×2反復＝80予定呼出し、各Builder40、40隣接paired blocks。concurrency1、automatic retry0。結果に応じた再抽出・追加生成・成功までretry・best-of-2を行わない。API失敗を新たな生成反復で置換しない。

SUMMARY_ONLY（S）は元JSONのsummaryだけ、FULL_RECORD（F）はsummary、building_type、shape、dimensions_estimate、materials、elements、rebuild_hints、uncertaintiesの8フィールドを抽出する。値・型・配列順・文字列を保存し、cohort_v4のstrict_saved_json（Decimal）と_serialize_exactで双方を同じsorted compact UTF8 JSONにする。最終入力には改行を追加しない。SもJSONでありproseではない。元にある誤り・unknown・曖昧さ・不確実性は修復しない。summaryを要約・翻訳・校正・切捨てせず、CLIPの72token制限を持ち込まない。GT、画像、旧応答、追加instruction、few-shot、repairを渡さない。

選択10件にある他のtop-level fieldsはprovider/model/building/created_atの管理情報で、両条件から除外した。未知フィールド0、欠損補完0。未知フィールドが出た場合は分類前にpayload作成を停止する。S/Fのdecoded summaryとUTF8 SHAは完全一致する。原description全体、summary、8-field typed tree、最終input、元O input IDをsource_binding_10とconditions_20_manifestへ束縛する。Fは旧O raw bytesのreplayではない。旧Oに含まれた管理情報を除き、S/F双方へ同じ再serializationを適用している。

管理情報、scene/condition/repeat ID、GT、過去IoU/CLIPScore、予算をモデル本文へ追加しない。ローカルmanifestだけに保持する。同じ条件の反復は同じpayload descriptorを参照する。10sceneの40固有payloadが80行で各2回使われる。SHAは保存payloadファイル、input本文、実transportのJSON bodyで区別する。

F−Sは他7フィールドをまとめて提供する方針の効果であり、文長、重複、矛盾、構造化の手掛かりも同時に変わる。paddingで長さを合わせない。各属性の独立効果、純粋な意味量、同一情報のprose/IR効果、自然captionerの情報損失量、言語化の不可避な損失を識別しない。FULL優位を前提にしない。

| Builder | exact request/response model | 設定 | endpoint |
| --- | --- | --- | --- |
| OpenAI | gpt-5-mini-2025-08-07 | reasoning.effort=medium; temperature/top_p/seed OMITTED; store=false; truncation=disabled; service_tier=default | Responses /v1/responses |
| Anthropic | claude-haiku-4-5-20251001 | temperature=1.0; thinking.type=disabled; top_p/top_k/seed OMITTED; service_tier=standard_only | Messages /v1/messages |

共通output cap32768。新request、stream=false、tools/web/images/会話状態なし、HTTP POST1回、redirect/retryなし、transport timeout600秒。長い生成のtimeoutはなお起こり得る。設定省略はtemperature0や決定性ではない。provider間の設定は同等能力を保証しない。未知response model、予期しないtool/thinking内容、未分類の不完全応答は成功にせず契約変更として停止する。

共通promptは `offline_evaluation/resmoke_v2/prompts/builder.txt`、SHA256 `ba7a555860628c1ad55ba5b65200ac3cf0d528be86c42f46dbb5bbdd61e6ccac`。変更しない。schemaは凍結resmoke_v2のoperations set/fill/carve。OpenAI text.formatのstrict json_schema、Anthropic output_config.formatのjson_schemaを使う。同一Builder内でS/Fのprompt/schema/sampling/cap/reasoningは同じ。旧adapter/MODELS/validatorを変更せず新専用adapterを使う。

公式資料の取得UTC・実bytes SHAと判断はPRICING_EVIDENCE.jsonに保存する。GPT5miniはdeprecatedで、停止予定は2026-12-11。Haiku4.5はActive (legacy)であり「not sooner than2026-10-15」を停止済みと解釈しない。アカウントavailabilityとnative schema acceptanceはNOT_RUN。停止／非対応なら自動fallbackをしない。将来の開始と各送信直前に取得後24時間以内を確認し、UTC日付変更だけでは止めない。期限超過は独立した同契約evidenceを追記する。価格・snapshot・対応条件変更は新たな明示承認を要し、旧planの黙った編集は認めない。停止日以降は新鮮な同契約evidenceでも送信を止める。

順序はscene→repeat→Builder→隣接S/F。各(scene,Builder)のrepeat1は独立domain付きhashの最下位bitでS/F順を決め、repeat2は逆順。Builder順もscene hashで決めrepeat2で反転する。全体でBuilder先頭回数を釣り合わせる。順序hashはdecoding seedではない。全80行はschedule_80.jsonに固定する。

費用はCOST_ESTIMATE.md/JSONに分離する。既存の全context入力予約（OpenAI400000、Anthropic200000 tokens）＋32768 outputを維持し、USD5へ合わせて縮めない。実input token数の主張はしない。最低でも次の隣接2呼出し分を送信前にdurable reserveし、known cost＋unknown/unsettled holdsがUSD5以内の時だけ進む。予算不足で止まる可能性を保持する。rawをfsync後に精算し、不明usageは全額hold、未送信と証明できないrequestは再送しない。401/403/520/timeout、HTTP応答有無、usage有無、model出力失敗を区別する。外部障害で止まった半ペアは明示する。

immutable disabled planはapi_allowed=false/budget_approved=false。準備中は実authority/approval/ledger/enabledを作らない。隔離fixtureの模擬承認は実承認ではない。将来のstartはcanonical plan、code/spec/payload/価格規約/費用scopeを表示し、/dev/ttyをreadとwriteの別handleで開きflushして完全一致入力を要求する。r+は使わない。拒否/Enter/EOF/read failure/nonTTYは未承認。prepared registryとcanonical lockでcopyの実行・二重起動・別rootによる台帳リセットを拒否する。resumeは同じ承認と台帳のNOT_SENTのみ。契約不一致後のresumeは禁止する。

各応答をローカル保存し、各建築完了時と安全停止時にcheckpointを作る。保存専用CLIは生成を呼ばない。新Draftだけを一覧＋Release IDで扱い、asset IDで取得、SHA/manifest/安全展開で検証する。tag-only404を不存在とみなさず、既存3Draftの公開/削除/上書きを行わない。LOCAL_VERIFIEDとGITHUB_VERIFIEDを区別する。保存失敗でも原本を残し、保存だけ再試行する。秘密をログやGitへ出さない。

評価対象は新80予定行だけ。旧OをFULLの結果に流用しない。GT voxel内容を読むのはCPU評価段階のprotocol-valid応答だけで、準備inputと送信adapterには渡さない。元GT descriptorをそのまま保持する。server_v2.2_strict_translationのoccupancy最適shiftを一度だけ選び、Goal010v2の凍結定義でsurface F1@1も同じshiftを使う。surface最適化、rotation、scale補正はない。

主幾何はaligned occupancy IoU、必須counter-metricはaligned surface F1@1。診断はoccupancy Precision/Recall、dimension MAE、material-aware IoU。正常emptyはprotocol-validのまま、定義されたIoU/surface0を保持する。正常成功の未定義値はNULL。全80行でcall identityとpayload/raw/text/parsed/voxel/metricsのhashを結び、未送信・未応答・評価成功・保存完了を分離する。完了応答率、protocol-valid率、非空率、truncation/validation/refusal/運用欠測の内訳と分母を残す。

主要差は各Builderで同じscene・repeatのF−S。IoUとsurfaceの双方について次の固定順で報告する。

1. completed-attempt utility：識別済み・usage既知のmodel-output failureだけ0を割り当てるbounded companion。正常成功は実値、operational unknownはNULL。正常emptyの0と失敗utility0を区別する。未知欠測があれば全予定入力への無条件効果ではない。
2. common-valid conditional accuracy：両条件ともprotocol-validかつ対象指標が定義されたペアだけ。treatment依存のvalid選択があるため、全入力の因果効果として単独で主張しない。

MAEへ失敗0を入れない。各sceneで適格な2反復の個別paired差を保存し、その平均をscene間で等重みに平均する。両反復あり／片方のみ／なし、各NULL理由、実paired数を保持する。S/Fの別々の独立平均の差やbest sampleに置換しない。2Builderは別々に報告し、独立建築Nは最大10。

探索的95% CIはplanned10scene frameからPCG64 seed20261017で2000 cluster draws。抽出多重度を保持し、同じpaired差drawを全指定estimandで共用する。元の寄与scene2以上、finite1900以上の時だけtype7 percentile CI、他はNULL。小標本、既存対象の再利用、有限drawに条件付けた区間を明記する。p値、効果／有意性による停止gate、新Holm family、CIを良くする再抽出は行わない。10×2反復の十分なpowerや精密な分散推定は保証しない。

採取後はまず技術的実行可能性、介入遵守、計測可能性、失敗とばらつきを判断する。Fが改善しても効いたフィールドは未識別。差が小さくても情報無用／同等／不可能とはしない。Fが悪化しても結果を保持し、元情報の誤り・矛盾・利用失敗を原因と断定しない。次の規模拡大は独立性・必要精度・費用を再設計して別承認する。CLIP相関、説明元比較、prose/IR、各フィールドablation、人手実験、論文を追加しない。

準備は既存依存だけを使い、CPU1200秒/RSS4GiB/newrun1GiB以内を確認する。保護inventoryは既存の直前検証を開始証拠として再利用しGoal014追加分を拡張、終了時に全保護集合を一度確認する。原本確認資源はfixture資源と分ける。旧60/960/934/25/HTTP520欠測1、961 attempts、991 calls、unknown holds2、global940承認retry、旧20比較・承認・予算・Draftは変更しない。
