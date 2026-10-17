**Goal013：summary と入力6視点の画像–テキスト整合度に関する採点前規約**

`POST_HOC_EXPLORATORY / GT_INDEPENDENT_SUMMARY_ALIGNMENT`。確認日：2026-10-17。

**現在は採点器を選択できず、実計算を実行しない。** 以下は候補の定義と、実行可能になった場合に守る解析規約である。この文書の存在は採点器の検証完了、CLIPScoreの算出、モデル取得の承認を意味しない。

| 現在の項目 | 状態・値 |
| --- | --- |
| scorer status | `BLOCKED_NO_VALIDATED_OFFLINE_SCORER` |
| selection status | `NOT_SELECTED` |
| active score_id / checkpoint | `null` / `null` |
| 実weight / tokenizer / scorer実装 SHA256 | すべて `null` |
| computation status | `COMPUTATION_NOT_PERFORMED` |
| 採点成功summary / 計画summary | `0 / 180` |
| 実image–text pair採点 / 計画上限 | `0 / 1080` |
| score-valid判定 | `NOT_ASSESSABLE_NO_SCORER` |
| correlation / concordance / CI / paired denominator | `null` / `NOT_COMPUTED` |
| RQ2_GATE | `BLOCKED_NO_SCORER` |

コードと仕様のcommit前の実得点計算・モデル推論・新しいoutcome結合は0。採点器不在の現状態ではcommit後もPhase 1・2へ進まない。合成fixtureの成功を実モデルの採点成功に数えない。旧Goal007〜012の研究結果は既知であり、これは盲検研究やpreregistrationではない。Goal013の得点・相関を見て採点法を選ぶことは禁止する。

透明性のため時系列を残す。機械的候補優先規則は11:06:30 UTCに記録し、可用性監査は11:09:30 UTCに候補0で完了した。その後、添付Goal011の取り違えを確認していた間にroot/agentsは既存Goal011のbaseline数値を再読した。新しいassociation計算はなく、採点0・候補0・未選択の状態は変わっていない。したがって「全担当者が過去outcomeに完全にblindだった」とは主張しない。正確な位置付けは、実採点前に規約を固定し、outcomeに応じたmodel変更を行っていない、というものになる。未実行の候補定義を選択済み採点法と扱わない。

**対象と主張の範囲**

対象は固定60建築×3 description providersの180保存O説明で、採点テキストは各原文JSONのトップレベル `summary` だけである。画像は保存された説明生成requestが指定した6視点を使う。このsource bindingはtransportによる実送信やモデルによる視認の独立証明ではない。元メタデータにある12候補画像を6枚のrequest画像と取り違えない。

これはO JSON全文の寸法・部品・材質等を網羅する品質評価でも、人間評定で妥当性確認された総合的な説明品質でもない。旧200件のDescription OverallやGoal008のGT由来proxyを代入しない。GT非依存とは、採点にGTボクセル・Builder出力・再構築指標を使わないという入力制約であり、3D理解・可視性・尺度の妥当性を保証しない。旧RQ2の判定は自動的に変更しない。

**可用性調査より前に記録した機械的選択規則**

規則原本は `/tmp/goal013-model-selection-rule.json`。SHA256は `0c12be725f7c01fe433265f2dda19a61a401e236c3c478c036b58a8b2d7aff75`。規則の作成時刻は `2026-10-17T11:06:30.752721+00:00` で、`selection_before_availability_inspection=true` を記録している。本規約作成時に当該ファイルのSHAを照合した。

1. 第一候補は正式CLIPScore。著者実装・識別可能な版、真正なOpenAI CLIP ViT-B/32 checkpoint、tokenizer/BPE、画像前処理、既知context、互換依存一式が既存ローカル環境に揃い、出典・SHA・ライセンスを検証できることを要する。
2. 第一候補が適格でない場合の第二候補は、真正な既存事前学習画像テキストモデルによる標準cosine similarity。checkpoint、tokenizer、前処理、context、cosineの定義、依存、出典・ライセンスが全て検証可能でなければ適格としない。正式CLIPScoreとは呼ばない。
3. 同順位に複数適格候補がある場合は、公表された参照実装・モデルの再現性、文書またはarchitectureから判断するCPU費用の小ささ、canonical checkpoint IDの辞書順の順に決める。費用比較のための推論benchmarkも行わない。

パッケージ名やcacheディレクトリだけから、必要な重みが揃うと推測しない。選択には実得点・GT・再構築outcomeを使わない。今回この規則で採用できるscorerはなく、**formal CLIPScoreは未採用の候補のまま**である。モデル・tokenizer・実装の未知SHAには `null` を使い、公開Web上の期待checksumをローカル実測SHAへ転記しない。

今後必要な資源が別途認められて揃っても、その事実だけでは実行しない。適格性を再監査し、唯一のscore ID・checkpointと全実装情報を非nullで固定した仕様・コードを、実採点より前の別commitに束縛する。選択後の自動fallback、相関を理由にしたモデル・template・context・hyperparameter変更は認めない。未承認の重み取得、install、既存venv/cache変更、研究/model/token-count/judge API、認証情報照会は行わない。

**候補の定義：正式CLIPScoreと標準cosineを区別する**

Hessel et al. (2021), *CLIPScore: A Reference-free Evaluation Metric for Image Captioning*, §3はViT-B/32、推奨prefix、cosineの非負化と2.5倍を定義する。[論文・書誌](https://aclanthology.org/2021.emnlp-main.595/)、[本文PDF](https://aclanthology.org/2021.emnlp-main.595.pdf)、DOI `10.18653/v1/2021.emnlp-main.595`。著者はJack Hessel、Ari Holtzman、Maxwell Forbes、Ronan Le Bras、Yejin Choi。15頁PDFの§3と評価定義を確認した。

画像特徴 `v` と、`"A photo depicts " + summary` のテキスト特徴 `t` を各々L2正規化したとき、正式候補の各視点値は `q = 2.5 × max(dot(v,t), 0)`。prefix末尾の空白も固定する。参照captionは使わず、RefCLIPScoreではない。式の理論的範囲は `[0,2.5]` で、1への上限clip、100倍、softmax、学習済みlogit temperatureは加えない。得点を校正された品質百分率とは呼ばない。

[著者実装](https://github.com/jmhessel/clipscore)の [clipscore.py](https://raw.githubusercontent.com/jmhessel/clipscore/main/clipscore.py) で `CLIPCapDataset`、`CLIPImageDataset`、`get_clip_score` を確認した。224-pixel bicubic Resize→CenterCrop→RGB→ToTensor→所定Normalize、prefix付きtokenizeの `truncate=True`、特徴正規化を再現する。CPU経路のfloat32と論文GPU経路のfloat16の違い、NumPy版による正規化差も著者実装が注意している。CPUで正式実装経路を再現しても、論文GPU値のbit完全再現とは報告しない。実装・dtype・ライブラリ版・前処理を検証できない変更版に正式名を付けない。

標準cosine候補は、採用checkpointの既定前処理と原summaryから得るL2正規化特徴の内積 `q = dot(v,t)` とする。追加prefix・clipping・2.5倍・temperature・softmaxは使わない。採用時の名称は実checkpointに即した `IMAGE_TEXT_COSINE_SUMMARY_ALIGNMENT_<checkpoint>_<version>` とし、CLIPモデルならその旨を明記する。これは現時点のactive score IDではない。

[OpenAI CLIP参照実装](https://github.com/openai/CLIP)の [model.py](https://raw.githubusercontent.com/openai/CLIP/main/clip/model.py) では `forward` がtemperature付きlogitsを返すため、それをraw cosineと取り違えない。[clip.py](https://raw.githubusercontent.com/openai/CLIP/main/clip/clip.py) と [simple_tokenizer.py](https://raw.githubusercontent.com/openai/CLIP/main/clip/simple_tokenizer.py) も確認した。名前指定のloadはdownloadへ進み得るので、実行可能な将来段階でも、hash検証済みの既存local checkpoint pathだけを使う。

両repositoryの公開ソース・README・MIT LICENSEはWeb読取りで確認したが、ソフト取得・install・実行はしていない。リンク先の `main` は可変で、本調査だけでは不変revisionを固定していない。実行前には実際のsource bytes、revision、依存を固定し、ローカルweightの由来とライセンスも別に確認する。CLIP原論文 *Learning Transferable Visual Models From Natural Language Supervision* (Radford et al., ICML2021) は[PMLR landing](https://proceedings.mlr.press/v139/radford21a.html)の書誌・abstractを確認した範囲に限り、同論文PDFを確認したとはしない。

6視点平均はGoal013の集約規約である。採用時は `MEAN6_<per-pair method ID>_SUMMARY_ONLY` と区別し、既報のmulti-view品質尺度として検証済みとは扱わない。

**Phase 0：source bindingと実行前固定**

各行で、building ID、description provider、元description/requestのbyte SHA、保存O input ID/input SHA、6画像の順序付きID/path/byte SHAを束縛する。保存descriptionとOのsummaryが同じ原文に対応することを検証する。不整合時は `SOURCE_BINDING_FAILED` で停止し、別画像・別summary・別建築への差し替えで埋めない。scene/provider間の6視点一致も監査する。GTを開いて採点入力の正しさを補強する処理は作らない。

summaryはJSONとしてdecodeした元の文字列をそのまま採点へ渡す。不存在・非文字列・空白のみは欠測判定にするが、判定のためのstrip結果を採点文字列に用いない。手動の短縮・翻訳・文章分割・言い換え・並べ替え、scene/provider IDやGT寸法の追記をしない。tokenizerに組み込まれたcleanupは固定実装の一部として記録する。

元JSONのファイルSHAとは別に、prefix付加前の `SHA256(UTF8(decoded summary))` を保存する。prefix付き文字列、実際のtoken ID列、画像transformにも別のprovenanceを持たせる。画像bytesは変更せず、checkpoint既定transformを採点前に固定する。実行device、dtype、batch size、thread数、乱数・決定性設定、全依存版も固定する。CPU完全決定性は検証できる範囲だけを記載する。

description生成のモデル版、時期、prompt、max_tokens等は元requestの交絡として保持する。既往監査のOpenAI/Anthropic 1800とGemini 4096はdescription生成予算であり、採点モデルのcontext上限とは別物である。

**Phase 1：summary限定の6視点採点と有効性gate**

各summaryを同じ6画像それぞれに対応させ、各視点値を `q_1,...,q_6` とする。primary scoreは `s = (q_1+...+q_6)/6`。同じ6値のmin、max、population SD `sqrt(sum((q_v-s)^2)/6)`、すなわち `ddof=0` を記述統計として保存する。画像特徴を平均してからcosineを計算する方式とは区別する。視点順序を変えても平均が変わらないことをfixtureで確認する。

全6視点が有限で、必要なsource/model hashが一致し、特徴normが非zeroの場合だけ、その行を `COMPUTATION_VALID` とする。1視点でも未計算・非有限・エラーならsummary得点は `null` とし、5視点等の残存平均を作らない。空summary、推論失敗、source/model不整合を0点にはしない。source/model不整合はrun全体の停止条件である。最高視点の選択、視点重み学習、providerごとの標準化はしない。

token diagnosticsは実tokenizerが検証された場合だけ計算する。OpenAI CLIPのcontext 77にはSOT/EOTが含まれ、payloadは最大75 tokensである。正式CLIPScore候補ではさらにprefixがその容量を消費する。summaryが77 tokens読めるとは記載しない。

同一tokenizerでprefixのpayload列 `P`、summaryの非切捨て列 `S`、結合文字列のpayload列 `T` を得る。`T = P || S` が実際に成立することを検証してから、summary retained countを `min(len(S), max(0,75-len(P)))`、dropped countを `len(S)-retained`、未読token率を `dropped/len(S)` とする。token境界が検証できなければ、文字数・単語数から代用せず `TOKEN_COVERAGE_NOT_VERIFIED` とする。標準cosineの追加prefixなしでは `P` は空。別checkpointなら実context・special-token規則を採点前の選択仕様へ固定し、77を流用しない。

原summaryのSOT/EOT予約文字列、cleanup後の空token列も検出する。黙って削除せず、実際のpooling位置と読取り量が検証できない行はPhase 2不適格とする。tokenizer自体が不在なら長さ・切捨て数・割合はすべて `null / NOT_COMPUTED`。原文の文字数をtoken数とは呼ばない。

各行に非切捨てtoken長、prefix長、special tokensを含む全長、実nonpadding長、retained/dropped count、未読率、切捨てflag、実token ID hashを残す。`COMPUTATION_VALID` と、`TRUNCATED`、`LOW_TEXT_COVERAGE`（未読率が0.5を超える）を別項目とする。

本規約では、**全体またはいずれかのdescription providerで、非空summaryの過半数の未読token率が0.5を超えた場合、`MEASURE_NOT_REPRESENTATIVE_OF_SUMMARY / INSUFFICIENT_MEASURE_VALIDITY` としてPhase 2を実行しない**。これは「大半を読めない」を停止へ結び付ける保守的な運用基準であり、外部検証された品質閾値ではない。token coverageが未確認の行があれば、除外件数と理由を明示する。非空summaryがないgroupの代表性は未判定とし、合格表示しない。

このgateを満たしても、部分的な切捨ては残り得る。長いsummaryだけを削除した有利なsubsetへprimaryを変えず、全適格行とprovider別のcoverageを報告する。gate通過を全summaryの意味理解や人間品質尺度の妥当性とは解釈しない。

180行の予定表には、source/O input/6画像/summaryの各hash、score ID、protocol commit、checkpoint・weight・tokenizer・実装hash、scoreとstatus、6視点値・min/max/population SD、token diagnostics、device/dtype/依存版を保存する。欠測行も削除しない。Phase 1の原本ファイルhashとメタデータを固定してから次段階へ進む。

**Phase 2：得点固定後だけ行う探索的関連**

現在は未実行であり、以下の統計値・実分母は `NOT_COMPUTED`。source gate、唯一のscorer、score artifact/hash、token代表性gateが揃って初めて、既存Goal007のO再構築outcomeを結合できる。結合keyはbuilding×description provider×O input/source identity。元のselected slotを1結果として扱い、attemptや承認retryを独立反復へ増やさない。

outcomeは保存済みOのaligned occupancy IoUに固定する。新alignment探索や指標選択はしない。alignedがGT参照の評価であることはoutcome側の性質であり、独立scoreへのGT入力を認める理由にはならない。Builderは `openai` と `anthropic` を分ける。

| primary探索 | 定義と分母 |
| --- | --- |
| A：pooled Spearman | 有限な有効summary得点とvalid O IoUの対応行をpoolし、平均順位同士のPearson相関を求める。OpenAI、Anthropic、両Builder同時validの同一説明に限るIoU算術平均の3表。 |
| B：建築内pairwise concordance | 各Builder別に、各建築の3 description providersから最大3つのunordered pairを作り、両得点・両IoUが適格かつ非tieのpairの順位一致率を求める。その建築内率を寄与建築間で等重みに平均する。 |

Aのtieは丸め前の保存値の完全一致で平均順位とし、providerごとに順位化・標準化しない。適格行が2未満、いずれかの順位vectorが定数、または算術が未定義なら推定値は `null`。寄与建築数・行数を併記し、少数で数学的に計算できることを十分な精度とはみなさない。Aは行をpoolするので、欠測により1建築当たりの寄与行数が変わる。cluster bootstrapを使っても点推定が建築等重みになるわけではない。

Bではまずmissing pairを除外する。両側観測済みのpairについて、得点のみtie、IoUのみtie、両方tieを排他的に数え、いずれもprimary concordanceの分母から除く。tieは丸め前値の完全一致で判定し、事後的なepsilon、乱択tie-break、0.5点を使わない。両差のsignが等しければconcordant、逆ならdiscordantとする。

建築 `b` の値は `C_b = concordant_b/(concordant_b+discordant_b)`。比較可能pairが0なら `C_b=null`。全体は比較可能pairを1つ以上持つ建築の `C_b` の算術平均であり、全pairのconcordant総数を総pair数で割った値に置き換えない。Bのprimaryは2 Builder別で、両Builder平均を新たなprimaryとして追加しない。

各表にplanned 60建築、適格description行数、寄与建築数を記載する。Bはさらに計画180候補pairs、missing、得点のみtie、IoUのみtie、両tie、比較可能、concordant、discordantの各件数を残す。AとBを同じestimandとは呼ばない。

valid-onlyをprimaryにする。`known_failure0` は別感度表で、元履歴が識別する完了済みmodel-output failureだけをIoU 0とする。独立scoreの欠測・推論失敗・正常出力の未定義scalar・運用欠測・unknownを0にしない。両Builder平均の感度表は、両側がvalidまたは識別済み完了失敗である同一説明だけを使い、primaryのboth-valid集合とは分ける。global942はJのHTTP520運用欠測であり、Oの失敗や得点欠測へ転用しない。元401と成功補完を独立観測に数えない。

**cluster bootstrap、欠測、CI**

固定60 building IDsを昇順にしたplanned frameから、各回60個を復元抽出する。`NumPy Generator(PCG64)`、seed `20261017`、2000 drawsに固定する。同じ抽出index行列を全指定estimand・感度表に共用し、各建築に属するproviderとBuilderの行をまとめて抽出する。2 Buildersを120独立建築、3説明を180独立建築とはしない。

同じbuildingが繰り返し抽出されたときは多重度を保持し、同じIDだからとgroupbyで1件に戻さない。Aは複製を含む各drawでpoolと順位を再計算する。Bは適格 `C_b` を抽出多重度付きで平均する。元frameの欠測建築を勝手に消してから60を引かず、各drawで適格な行・建築コピーだけを分母に入れる。

未定義drawは `null` として数え、代わりのdrawを補充しない。元の寄与建築が2以上で、2000 draws中1900以上が有限の場合だけ、有限drawの2.5/97.5 percentile（type7、NumPy `linear`）を探索的95% CIとする。それ以外は点推定・有効/無効draw数を保存し、CIを `null / INSUFFICIENT_BOOTSTRAP_VALIDITY` とする。有限drawに条件付けた区間であることも明記する。1900という基準は運用上の安定性gateで、有意性判定ではない。

p値、significance label、Holm family、CIを見ての再抽出回数増加、有利なsubsetへの変更を行わない。探索的CIはcaption品質尺度の妥当性や因果性の証明ではない。

**固定する補助診断と終了条件**

各description providerの推定可能性、行・建築の実分母、失敗による分母差、summary長・切捨て分布を報告する。得点と非切捨てsummary token長のSpearmanは全provider pooledとprovider別の補助診断に限り、同じcluster規約・tie/欠測規約を使う。6視点min/max/SDはview変動の記述に留める。新しいscene難度指標、GT由来の可視性mask、長文分割、重み学習、GT proxyとの追加探索はこの固定primaryに持ち込まない。

将来scorerが適格な場合の検証は、source/hash改変拒否、別model拒否、欠測summary、1視点失敗時のNULL、6視点順序に対する平均不変、同文字列・同画像のreplay、画像入替、同一cluster重複とtie処理に絞る。実モデルを使わないfixtureだけでは画像テキスト採点の妥当性を検証したことにならない。

現状はscorer不在なので、tokenizerによる実token測定、1080 pair推論、得点集約、outcome結合、Spearman、concordance、bootstrapを全て停止する。数値欄を0で埋めず、`null / NOT_COMPUTED` を維持する。モデル取得・install・別モデルへの無断切替によってこの停止を迂回しない。既存のsource binding調査と不足要件の記録だけで終了する。

将来得点が得られても、許される問いは「原summaryの画像–テキスト整合度surrogateが、保存Oの再構築IoUとどの程度関連するか」までである。無相関・負相関は「説明品質一般が再構築可能性を表さない」の証明にならず、正相関も説明品質が下流品質を決めるという因果結論にはならない。画像/建築難度、provider、モデル学習分布、summary外のO情報、出力失敗の選択が残る。原RQ2・RQ3・RQ4を検証済みへ変更しない。
