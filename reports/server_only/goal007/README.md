# Goal007：最終統合解析結果

分析ラベルは **fixed60 analysis with one operational missing observation**。
既存の固定60建築・960入力をCPUで統合した。全960件の完全測定ではない。
このGoalの研究API送信、モデル照会、認証キー照会はすべて0回。

## 実台帳による分母

| 項目 | 件数 |
|---|---:|
| 固定スロット | 960 |
| 固定対象の試行履歴 | 961 |
| 歴史的calls（先行smoke30件込み） | 991 |
| 選択した確定完了応答 | 959 |
| 厳密評価のvalid観測 | 934 |
| モデル出力失敗 | 25（truncation19、validation6） |
| 運用上の未測定slot | 1（global942、HTTP520） |
| 運用失敗attempt | 2（元940の401と942の520） |
| bounded IoU failure0適格slot | 959（valid934＋既知モデル失敗25） |
| unknown usage予約 | 2 |

各建築はO=6、J=6、P=2、E=2。O/J各360、P/E各120で、oracleを説明文provider別に複製していない。
global940は元401のattemptと予約を保持し、同一payload・承認起源を確認した補完成功を1観測として選択した。
global942（building_085、J、OpenAI Builder）はusage/response/metrics=nullのまま。IoU=0やモデル失敗にしていない。

未知usageの予約は各USD0.165536、合計USD0.331072を保持した。
既知usageによる保守的計算USD21.22976425と予約を合わせた予算計上は **USD21.56083625**。実請求額は不明。
詳細は[data/cost_validation.json](data/cost_validation.json)、[attempts](data/attempts_audit.csv)、[slot mapping](data/source_to_slot_mapping.csv)。

## 実測値と解釈

以下はvalid観測を建築内で平均した後、建築を等重みで平均した値。IoUは高いほど良く、寸法MAEは低いほど良い（blocks）。

| Builder | 条件 | valid / planned slots | valid建築数 | aligned IoU | 寸法MAE |
|---|---|---:|---:|---:|---:|
| OpenAI | O | 177 / 180 | 60 | 0.1760 | 12.8259 |
| OpenAI | J | 177 / 180 | 60 | 0.1712 | 13.1639 |
| OpenAI | P | 60 / 60 | 60 | 0.1674 | 4.4000 |
| OpenAI | E | 55 / 60 | 55 | 0.9219 | 0.3758 |
| Anthropic | O | 179 / 180 | 60 | 0.1772 | 13.3870 |
| Anthropic | J | 180 / 180 | 60 | 0.1754 | 13.5648 |
| Anthropic | P | 60 / 60 | 60 | 0.1691 | 0.0444 |
| Anthropic | E | 46 / 60 | 46 | 0.9919 | 0.0362 |

Oのabsolute IoUはOpenAI0.0762、Anthropic0.0874で、平行移動による位置合わせ後も上表の水準にとどまる。
Oのaligned material-aware IoUは0.0210／0.0169、normalized repair costは1.4888／1.8732。
この固定条件では配置だけでなく形状・材質の再構築にも誤差が残る。
Pは寸法誤差を減らすが、P−OのIoU差にはHolm補正後の有意性を認めない。
Eはvalid出力の幾何を改善する一方、OpenAI5件・Anthropic14件のtruncationを含む。
Eのfailure0補助IoUは0.8451／0.7604であり、valid-onlyと異なる問いに答える。

事前指定3family（8/4/8比較）、seed20261013、建築bootstrap2,000、両側sign-flip10,000、family内Holmを変更していない。
代表的な共通validペア比較は次の通り。全20比較とraw pは[comparisons_20.csv](results/comparisons_20.csv)に掲載した。

| 比較 | 指標 | pairs / 建築数 | 平均差 [95% CI] | p_holm |
|---|---|---:|---:|---:|
| Anthropic−OpenAI、O | IoU | 176 / 60 | +0.000593 [−0.005610, +0.006937] | 1.0000 |
| Anthropic−OpenAI、P | MAE | 60 / 60 | −4.355556 [−5.450000, −3.383333] | 0.00079992 |
| J−O、OpenAI | MAE | 175 / 60 | +0.335185 [+0.113843, +0.562963] | 0.01279872 |
| E−O、OpenAI | IoU | 55 / 55 | +0.745755 [+0.686232, +0.801694] | 0.00079992 |
| Anthropic−OpenAI、E | IoU | 46 / 46 | −0.002797 [−0.010450, +0.006185] | 1.0000 |

特にEの単群平均は55建築対46建築で分母が異なる。共通46建築の比較からAnthropicのIoU優位は示されず、単群平均の差で優劣を述べない。
JはOの情報保存型JSON再シリアライズであり、旧Structured IRではない。P/Eの差はGT追加を含む診断で、純粋な言語情報損失の因果効果ではない。

**RQ2: NOT_TESTABLE_WITH_CURRENT_MATCHED_SCORES**。
旧per-sample値は存在するが、現行v4の説明文原本との厳密な対応を証明できず、Claudeの一部には内容不一致、Geminiには対応スコア不足がある。
旧モデル平均3点やv5を流用した相関は計算していない。[照合監査](rq2_matched_scores_audit.json)を参照。

## 原本とGitHub復元

元validatorによる4sourceの確認と、前後で5,811ファイルおよびディレクトリinventoryの不変を確認した。
14既存lockは読取り専用で取得。生成・uploadプロセスとの競合はなかった。
3 DraftはRelease ID/asset IDで各6 assetsを新規取得し、size/digest/SHA、安全展開、ledger/claims/responses、参照連鎖、copied plan起動拒否を検証した。

| Draft Release ID | 原本 | 6 assets | 再取得・安全復元・原本照合 |
|---:|---|---:|---|
| 414157245 | Goal006・781試行 | 6 / 6 | PASS |
| 414396750 | Goal006B1・3試行 | 6 / 6 | PASS |
| 414429861 | Goal006C・17試行 | 6 / 6 | PASS |

既存Draftの公開・編集・削除・置換は0回。[audit](audit/audit_gate.json)と[MANIFEST](MANIFEST.json)に全source、identity、hash、release/asset IDを保存した。
原本rawを報告ディレクトリへ複製していない。57解析ファイルは合計17,158,720 bytes、最大単一ファイル7,653,698 bytes。
全ファイルが10MiB以下であるため、新規Draftは不要。専用ブランチへ小規模なコード・解析表・図・証跡を保存する。

## 成果物

- [960行JSON](data/final_slots_960.json) / [CSV](data/final_slots_960.csv)、[961試行](data/attempts_audit.csv)、[検証](data/data_validation.json)
- [全統計JSON](results/results.json)、[群別320行](results/per_group.csv)、[20比較](results/comparisons_20.csv)、[全分母](results/denominators.json)、[942感度分析](results/sensitivity_missing942.json)
- [失敗分類](data/failure_taxonomy.csv)、[モデル出力失敗](data/model_output_errors.csv)、[運用失敗](data/operational_errors.csv)、[会計](data/cost_validation.csv)
- LaTeX：[主表](results/table_main.tex)、[条件差](results/table_ablation.tex)、[失敗](results/table_failures.tex)、[統計](results/table_statistics.tex)
- 図：[主結果PNG](results/figure_main.png) / [SVG](results/figure_main.svg)、[説明文×Builder PNG](results/figure_description_builders.png) / [SVG](results/figure_description_builders.svg)、[全比較PNG](results/figure_comparisons.png) / [SVG](results/figure_comparisons.svg)
- [英語Results/Discussion/Limitations](results/ARR_RESULTS_EN.md)、[解釈補足](INTERPRETATION_EN.md)、[日本語要約](results/SUMMARY_JA.md)、[方法・再現監査](results/AUDIT.md)
- [図の数値出典](results/figure_sources.json)、[描画成果物hash](results/render_artifacts.json)、[公開コピー照合](publication_copy_verification.json)

## CPU検証・再現

最終解析コードcommitは `7500cea2160a8489f0e8bd06b6249d261226f943`。
開始HEADは `c2b7ab848ade1e86bdc125064de000579e727783`。branchは `research/server-only-20261013`。
mainは `7309327efd1c19dda793b19e8d8e6ce0b5868db9` のまま。
環境はroot、Ubuntu24.04.5、Python3.12.3、workspace-write／auto_review。通常コマンドと対象別承認を使用し、OS隔離の強度は保証していない。
描画依存は別venvへ導入し、既存venvとOSを変更していない。[環境](environment.json)、[依存固定](../../../offline_evaluation/analysis/goal007/plot_requirements.lock)。

最新の固有テストは **88 PASS / 0 FAIL / 0 ERROR / 0 SKIP**。
内訳はaudit31、statistics28、integration13、CLI5、renderer11。修正前CLI4件を二重計上せず、他Goalの件数を加えていない。
統計→rendererの別プロセス接続検証も合成fixtureでPASS。[テスト証跡](final_validation.json)。
初回実解析はMANIFEST保存時のPath型エラーで終了5となった。出力を保全し、回帰テスト・修正commit後に別出力先で完走した。
初回と最終の統計JSONは同一で、初回失敗の証跡も[保存](initial_analysis_stop.json)している。

最終実解析は終了0、127.34秒（内部計測127.03秒）、最大RSS193,932KiB（描画子プロセスを含むtime計測）。
内部親プロセスRSSは127,248KiB。終了時空き容量69,630,095,360 bytes。
strict評価915件を再利用、補完19成功出力を新規評価、resource limit0。初回停止runでも19件を評価したため、Goal全体では19対象を各2回CPU評価した。元915件の再評価・旧cache書込みは0回。
PNG/SVGとLaTeXソースは数値・表示を照合済み。TeXコンパイラがないためPDFコンパイルは未実施。

同じサーバーで、存在しない新規出力先を選び、repo rootから実行する。原本と監査gateが不一致なら停止する。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv-server-v2-20261013/bin/python -B -m offline_evaluation.analysis.goal007.cli \
  --audit-gate /root/minecraft-runs/goal007-analysis-20261016T130420Z-a8a7a399/audit/audit_gate.json \
  --destination /root/minecraft-runs/goal007-analysis-replay-CHOOSE_NEW_UTC \
  --code-commit 7500cea2160a8489f0e8bd06b6249d261226f943 \
  --plot-python /root/workspace/Minecraft-view-dataset/.venv-server-v2-goal007-plots-20261016/bin/python \
  --rq2-audit /root/workspace/Minecraft-view-dataset/reports/server_only/goal007/rq2_matched_scores_audit.json
```

原本解析先：`/root/minecraft-runs/goal007-analysis-20261016T130420Z-a8a7a399/analysis-final`。
最終MANIFEST SHA256：`ef05c2ed93247802e9a68a00d700d39a8934d6231c7787315ab37b01d2467714`。
最終push後のcommit/remote/main/clean確認は同run rootの新規 `publication_receipt.json` と完了報告に記録する。

## 投稿に向けた限界とAPIなしの優先作業

独立Nは最大60で、保存説明文・モデル・prompt・GTが固定された実験である。valid選択、25出力失敗、1運用欠測、承認retryの選択、RQ2対応不足により一般化範囲が限られる。
942のIoU範囲0–1は探索的な可能値boundであり、観測値や新しいp値ではない。非有意差は同等性を証明しない。

1. 既存の材質・repair・truncation内訳と出典付き再構築例を整理し、RQ1の誤差構造を明示する。
2. 共通validとfailure0の分母、Eの選択、942感度、全paired CIを論文の主張に反映する。
3. 保存済み説明文と品質スコア原本の厳密な対応証拠を追加探索する。対応が得られなければRQ2は未検証とし、新たなCPU品質評価は別の探索的プロトコルとして定義する。
