Goal015の選択前固定規則。保存済み結果は既知であり、盲検・独立外部検証・preregistrationではない。対象の選択を実行する前にこの文書をcommitする。

元evaluation60の保存済みsize_rank_stratum（5層）を継承する。各層で `SHA256(UTF8("goal015-scenes-v1\0" + scene_id))` の昇順、同hash時だけscene_id辞書順で先頭2建築を選ぶ。結果・CLIPScore・過去IoU・成功失敗・材質誤り・例示sceneは使わない。source欠落/不正でも置換しない。

各層の選ばれた2sceneを `SHA256(UTF8("goal015-presentation-v1\0" + scene_id))` とscene IDで並べ、1件ずつ2巡へ配置する。各巡の層順は `SHA256(UTF8("goal015-strata-order-v1\0" + str(round_index) + "\0" + str(stratum)))`、同hashはstratumで決める。round_indexは1,2。この順序で前半・後半に各層1件ずつ含まれる。

sceneごとrepeat1のBuilder順を `SHA256(UTF8("goal015-builder-order-v1\0" + scene_id))` の整数値の最下位bitで決める。bit0ならopenai→anthropic、bit1なら逆。repeat2は逆順とし、全体で各Builderが先頭の回数を釣り合わせる。scene→repeat1,2→Builderの順にpaired blockを置く。

各(scene,Builder)のrepeat1条件順を `SHA256(UTF8("goal015-condition-order-v1\0" + scene_id + "\0" + builder))` の整数値の最下位bitで決める。bit0ならSUMMARY_ONLY→FULL_RECORD、bit1なら逆。repeat2は逆順。各S/F paired blockは隣接2呼出し、合計40blocks/80calls。hashはdecoding seedではない。

保存OpenAI説明だけを用い、summaryと他7フィールドの追加方針を比較する。80行を選んでから生成結果に応じた選び直し・順序変更・成功までretry・旧FULL出力の流用・best-of-2はしない。研究API・model/token-count/key queriesは今回0。planはapi_allowed=false、budget_approved=falseとし、将来の実TTY承認とは区別する。
