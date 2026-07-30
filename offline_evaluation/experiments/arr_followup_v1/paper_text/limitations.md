# Limitations

Although the structured representations were generated solely from their corresponding free-form descriptions and passed automated field-level provenance checks, we did not conduct a human semantic equivalence study. Subtle changes in ambiguity, emphasis, uncertainty, or spatial interpretation may therefore remain.

Our reconstructions are evaluated through a deterministic offline voxel executor rather than embodied agent execution in Minecraft. We do not model Minecraft physics, navigation, reachability, block-support constraints, interaction sequences, or execution failures caused by an agent's viewpoint and movement. Consequently, the results characterize discrete voxel reconstruction under the repository action semantics, not end-to-end Minecraft task completion.

The experiments use closed models whose behavior may drift despite fixed model identifiers, prompts, and logged response metadata. The Gemini description cohort was regenerated under a unified 2026 condition and may not be directly interchangeable with earlier closed-model outputs. Most controlled reconstruction results use a single OpenAI fixed builder, so representation effects may depend on that builder's capabilities and prompt interpretation.

The stochastic analysis uses a stratified subset rather than all 200 buildings. The Direct Image-to-Build analysis likewise uses a stratified 100-building subset and should not be generalized to the full benchmark without qualification. Direct and language-mediated conditions share the gpt-5-mini model identifier, validator, executor, and action schema, but require modality-specific prompts; thus, differences cannot be attributed solely to the presence or absence of textualization.

Finally, we did not evaluate reconstruction by human builders and make no claim that the observed effects generalize to human interpretation or construction.
