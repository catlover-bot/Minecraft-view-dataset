Goal014は固定した12項目の診断を完了した。Goal013Aの結果を知った後のPOST_HOC_EXPLORATORYであり、元score・元CI・原RQ2を変更しない。

Q-Aでは、このsummary集合に生成元画像を他建築の画像から区別する信号が見られる。Dは0.03372 [0.02802, 0.03944]、Wは0.82321 [0.77532, 0.86436]。Wは59の他建築より対応建築の得点が高い割合（完全tieは半点）の建築等重み平均で、top-1正解率ではない。D=0とW=0.5は無情報モデルの参照値であり、事後的な合格基準を作っていない。非対応画像は文章内容が確実に偽になる負例ではない。

Q-Bのsurface F1@1とのrhoはOpenAI0.02015、Anthropic0.02519で、どちらの区間も0を含む。両Builderともcommon-valid集合は元IoU集合と同じ175/177行・60建築。Anthropicの同じdraw内の差delta_rhoは0.14908 [0.02518, 0.28323]で、元の負のIoU関連よりsurface側の係数が高い方向を示す。ただしsurface自体の明瞭な正の順位関連を示す結果ではない。OpenAIの差は0.09707 [-0.03352, 0.23132]。異なる集合や別々のCIの比較ではない。

Q-Cでは未切捨て集合でも係数はOpenAI−0.16671、Anthropic−0.21880で、切捨てを除くことで明瞭な正の関連へ変わる結果ではなかった。未切捨ての実paired Nは118/119、切捨ては57/58。これはprovider・文長・内容・建築難度の構成も変える選択なので、群差から切捨ての因果効果は識別できない。未切捨て群をprimaryへ昇格しない。

| 問い | Builder／群 | 推定項目 | 実paired記述行 | 寄与建築 | 推定値 | 探索的95% CI |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Q-A | 全体 | D_i | 178 | 60 | 0.03372 | [0.02802, 0.03944] |
| Q-A | 全体 | W_i | 178 | 60 | 0.82321 | [0.77532, 0.86436] |
| Q-B | openai | rho_iou | 175 | 60 | -0.07692 | [-0.22561, 0.06731] |
| Q-B | openai | rho_surface | 175 | 60 | 0.02015 | [-0.13532, 0.16903] |
| Q-B | openai | delta_rho | 175 | 60 | 0.09707 | [-0.03352, 0.23132] |
| Q-B | anthropic | rho_iou | 177 | 60 | -0.12389 | [-0.26488, 0.01379] |
| Q-B | anthropic | rho_surface | 177 | 60 | 0.02519 | [-0.12304, 0.17179] |
| Q-B | anthropic | delta_rho | 177 | 60 | 0.14908 | [0.02518, 0.28323] |
| Q-C | openai／未切捨て | rho_iou | 118 | 59 | -0.16671 | [-0.33050, -0.00789] |
| Q-C | openai／切捨て | rho_iou | 57 | 44 | 0.03669 | [-0.22037, 0.29014] |
| Q-C | anthropic／未切捨て | rho_iou | 119 | 59 | -0.21880 | [-0.39854, -0.03583] |
| Q-C | anthropic／切捨て | rho_iou | 58 | 46 | -0.05829 | [-0.29269, 0.17366] |

全12項目が2000/2000 finite、undefined0。元点推定・寄与建築数・finite1900以上のCI gateを満たし、type7 percentile区間を保存した。全項目で同じplanned60建築のPCG64 seed20261017のdrawを共有し、多重度を保持した。Q-Aは建築等重みでgallery固定に条件付けたquery建築の不確実性。Q-B/Cは記述行等重みpooled係数で、cluster化はCIの依存構造を扱う。新しいp値・significance label・Holm family・equivalence検定はない。

切捨てなし119件のprovider件数はOpenAI22/Anthropic46/Gemini51、token長min/median/maxは38/63/72。切捨て59件は38/14/7、73/80/127。各Builderの実paired集合の件数・token記述と全row ID/maskはRESULTS_SUMMARY.jsonに保持した。追加provider相関・length補正・回帰は行わない。

識別できたのは、固定galleryへの来歴上の対応を検出する信号と、保存Oの2種類の幾何outcomeへの限定的関連、および固定truncation別の傾向である。Q-Aの良好な識別は個々の細かな正誤・空間関係・人間品質を検証しない。surface F1@1は6近傍boundary voxel中心の距離で空洞内壁を含み、元occupancy最適shiftを継承する。可視画像外観・mesh表面・意味的間取りの評価ではない。

実装parity PASS、固定gallery識別信号、Minecraft人間品質妥当性未検証、RQ2限定版、原RQ2未検証を区別する。BuilderはJSON全文を読みscorerはsummaryのみなので、これらの係数から全文品質一般・独立外部予測・情報損失の因果効果は分からない。GT非参照も統計的独立を意味しない。0を含むCIは関連不存在の証明ではなく、Q-Aの結果も外部cohortへの保証ではない。

元Goal013Aと既定failure0感度は凍結したまま参照する。新failure0表、both-Builder平均表、他尺度、モデル・prefix・crop変更、追加生成、論文、次Goalは作成しない。この12項目の診断で終了する。
