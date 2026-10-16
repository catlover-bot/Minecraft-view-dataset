# 説明生成元の探索的比較

POST_HOC_EXPLORATORY。全36対比を保存した。同じ建築・Builder・O/J条件のcommon-valid pairのみを比較し、1建築1差の等重み平均を示す。差はAnthropic−OpenAI、Gemini−OpenAI、Gemini−Anthropic。IoUは高いほど、MAEは低いほど良い。

bootstrap frameは各対比のcommon-valid建築であり、baselineのplanned frameとは異なる。2000 building drawsと95% percentile CIを使用し、新しいp値は計算しない。モデルversion・実画像・prompt・出力上限・時期を含む観測条件の交絡がある。固有能力や純粋な表記変更の因果効果を主張しない。全pairs・除外理由・NをJSON/CSVに残し、2 BuildersのNを合算しない。
