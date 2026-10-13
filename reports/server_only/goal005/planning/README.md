# Goal005 固定60評価・第1バッチ準備

追加研究API送信0。`manifest.json` は未承認の入力計画であり、enabledや送信予約ではない。Goal002の固定10 smoke・60 evaluation・30 restをそのまま再利用した。未見データとは呼ばない。060/075はsmokeに残り、評価60へ入れていない。

- 固定cohort原本: `/root/minecraft-runs/goal002-20261013-141854/cohort.v4.json`
- 原本SHA256: `4015da18a189b597365d412fe21686139c62c809e4349e917b9ee4b971548ff0`
- 旧両provider入力480件の同一ファイルSHA256: `6b4d7ce9e08289940f0dd6b7d95cc396e6f11219e9c8bd63b306d99d845cc846`
- 新manifest identity: `7dbf373a7a239329bad50bfe2bf5340bced737534ee1623c05282207299e990d`

元cohortの100件のsplitとsource manifest hashを保持し、評価60のGT・保存説明・画像参照を含むsource SHAを実ファイルへ照合した。旧入力manifestとの整合性も検証した。O/J/Pは旧凍結本文と完全一致し、Eだけ旧Z-runを現行resmokeのcuboid表現へ置換した。生成prompt/schema/output cap32768/Validator/Executor/metricは現行resmokeと同じ。旧960を有効化していない。

## 選択と順序

各既存stratum内で保存済み`stratum_draw_order`、次にscene ID順とし、各順位をstrata 0→4の順に交互に取る。最初の2周が第1バッチで、各stratumから2建築になる。結果や応答長を選択へ使用していない。

第1バッチの実行順は **087, 026, 007, 032, 092, 078, 045, 023, 055, 062**。各建築のsemantic inputはO/J×保存説明3モデルとP/E各1の8件、各入力にOpenAI→Anthropicを交互に割り当てる。semantic順は固定の`O/openai,J/openai,O/anthropic,J/anthropic,O/gemini,J/gemini,P,E`を建築順序番号 modulo8だけ回転して条件位置の偏りを抑える。全960入力の最初の160件が第1バッチであり、残り800は未承認。

`input_manifest960.csv` に全960のinput ID・本文hash・source hash・旧入力ID/hash・stratum・順序を記録。各providerの条件数はO180/J180/P60/E60。第1バッチは各provider O30/J30/P10/E10。

## CPU長診断

今回必要な評価60のみを計測。全60でcuboid CPUデコードと参照action Executorが位置・材質ともGTへ完全一致した。これはモデルの成功ではない。既存200GTや旧1200の全再評価は行っていない。

既存`/tmp/goal004-tokenizer-20261013`の **tiktoken0.14.0 / o200k_base** と既存語彙cacheを使用し、依存取得・download・token-count APIは0。語彙SHA256は`446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`。CPU計測11.501315秒。全480semantic inputの本文byte/token、prompt/schema込みのローカルtext token診断を`input_lengths.csv`へ記録した。最大は29,905で、OpenAIの公表context400,000とcap32,768に対して局所text計測は収まる。ただしprovider framing・schema注入・実LLM出力長・Anthropic正確token数は保証しない。

**9/60建築の参照action本文が32768 local tokensを超える**。第1バッチにも032と092が含まれる。

|建築|cuboids|参照action local tokens|
|---|---:|---:|
|032|1309|51111|
|092|874|34574|
|094|1247|48655|
|049|2046|79978|
|077|903|35732|
|037|895|35626|
|001|1545|60690|
|088|897|35291|
|085|1282|50208|

参照列は一つの実行可能なCPU構成で、最短列やモデル出力長の下限ではない。現行計画は全60とcap32768を維持し、切詰めが発生した場合は資源制約下の観測結果として保存する。大きな建築の除外や本文切詰めは0。cap拡張・表現変更は別条件の設計変更になり、利用者の明示判断と承認が必要で、同一固定条件としてpoolできない。

## 160/960への実測外挿と保守的予約

保存済み16応答のprovider×条件それぞれ2建築のinput/output tokens・台帳費用上限・開始終了時刻を使用し、建築当たりの条件重みを**O3/J3/P1/E1**にした。失敗2の実測消費も含めた。各16観測は`empirical_usage16.csv`、8群の平均とprovider別外挿は`costs.json`にある。

|計画|入力tokens外挿|出力tokens外挿|実測費用外挿USD|実測時間外挿|全context+cap予約tokens|全context+cap予約USD|
|---|---:|---:|---:|---:|---:|---:|
|10建築160入力|427845|998720|3.1722475|7855.12182秒（約2.18時間）|53242880|42.350080|
|60建築960入力|2567070|5992320|19.033485|47130.73092秒（約13.09時間）|319457280|254.100480|

外挿は2建築しかなく、O/Jの実測説明はOpenAI由来のみ。残り説明モデル・建築規模・出力長・reasoning・待ち時間の差を含む不確実性が大きい。時間はattemptの開始から終了で、CPU集計・保存・中断時間を含まない。費用は保守的台帳値で請求確定額ではない。confidence intervalや完了保証は付けない。

予約額は公表contextの全量を入力として確保し、output capを別途加えた保守的合計であり、実測外挿とは別物。提案する**新旧合算USD10は未承認**で、旧消費USD0.68679325を差し引いた残額が1送信分の予約を満たさなくなれば未送信で停止する。累積calls上限190、retry上限4（追加0）。160完了費用の予測でも、残り800の実行承認でもない。

## 再現

`offline_evaluation.batch_v1.inputs`の`make_manifest`→`length_report`→`extrapolate`→`save_reports`を、存在しない新規出力先に対して実行する。`length_report`には既存ローカルtokenizerを指定した環境 `PYTHONPATH=/tmp/goal004-tokenizer-20261013 TIKTOKEN_CACHE_DIR=/tmp/goal004-tokenizer-cache-20261013` を使用した。cache missはdownloadせず例外で止まり、今回の計測中はsocket.connectも禁止した。料金入力は`reports/server_only/goal005/official_pricing.json`、実測源は`goal004a-ttyfix-20261013T194600Z/{plan.json,enabled/ledger.json}`。鮮度の一致は過去のCPU読取りの条件にはしない。

入力順序・固定split・改変source拒否・両provider共通本文・正しい条件重み・不明usage拒否を検証する14 unit testsがPASSした（0.21秒）。
