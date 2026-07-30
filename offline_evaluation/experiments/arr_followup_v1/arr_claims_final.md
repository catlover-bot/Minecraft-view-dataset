# Final ARR claims

## Recommended RQs

1. Which properties of generated descriptions predict downstream reconstruction performance when builder capability is controlled?
2. How does description-only structural canonicalization, with automated provenance auditing and documented unsupported fields, affect geometric, material, and positional reconstruction fidelity?
3. Where does transfer from attribute recognition to discrete spatial grounding break down?
4. On a stratified 100-building subset, how does direct image-to-build generation compare with language-mediated reconstruction using the same model family?

Here, content-preserving is limited to provenance: generated solely from the corresponding free-form description, without access to images, ground truth, or inserted default values. The strengthened audit identified unsupported duplicated top-level fields in a subset, so perfect field or semantic preservation is not claimed and a strict-supported sensitivity analysis is reported.

## RQ answers

RQ1. Description Overall moderately predicts absolute and aligned geometry; dimension and material-description scores predict the corresponding explicit attributes more strongly than exact spatial grounding. These are associations under one controlled builder, not causal effects.

RQ2. Across 600 pairs, canonicalization reduced Absolute Occupancy IoU by -0.0095 (95% CI [-0.0134, -0.0057]) while reducing bbox dimension MAE by 0.2228 blocks. It did not yield a robust aligned-IoU improvement; increased centroid displacement and diagnostic position-grounding loss are consistent with the absolute-IoU decline.

RQ3. Attribute identity transfers better than absolute grounding. Position-grounding loss increased by 0.0138, and centroid distance increased by 1.067 blocks under canonicalization, while material selection remained much stronger than exact position-and-material agreement.

RQ4. Direct image-to-build used the same gpt-5-mini model family on a stratified n=100 subset. It achieved higher aligned geometry and lower bbox error but lower Absolute Occupancy IoU, showing that direct visual access did not solve coordinate anchoring; results do not support a uniformly superior direct pathway.

## Supported claims

- Canonicalization improved bbox dimensional fidelity but reduced absolute positional occupancy fidelity in n=600 paired observations; exact CIs and Holm-adjusted tests are in the parent and follow-up tables.
- The Absolute-IoU decline is associated most strongly with increased centroid displacement in the building-cluster-robust regression; it is not explained by an aligned-geometry decline.
- Completeness remained positively associated with the original GT-normalized repair cost after model, GT size, bbox volume, material count, and description length adjustment (coefficient 1.352, 95% CI [0.780, 1.924], p=3.619e-06, n=600).
- Direct image-to-build improves some intrinsic/dimensional metrics but not absolute coordinate or material grounding on the stratified n=100 subset.

## Partially supported claims

- Automated provenance supports the absence of GT/image/default insertion, but 41 Gemini pairs contained 205 unsupported duplicated top-level fields and therefore require a strict-supported sensitivity analysis. The audit does not prove semantic equivalence.
- Direct results are subset-specific and prompt-modality differences remain even though the model ID is shared.
- Model-specific representation effects and stochastic stability must be interpreted using their per-model and per-run tables rather than pooled trends alone.

## Unsupported claims

- Structured IR generally improves reconstruction.
- Structured IR recovers missing information or is perfectly semantically equivalent to Free-form.
- Results generalize to human builders, Minecraft agent execution, physics, navigation, or reachability.
- Aligned IoU measures absolute positional accuracy.
- Direct Image-to-Build is uniformly superior.
- Textualization causes information loss.
- Correlation proves causal influence.

## Recommended central message

Generated descriptions preserve explicit dimensional and material identities substantially better than their discrete spatial grounding. Description-only canonicalization modestly improves bounding-box dimensions but does not resolve coordinate or material placement and can reduce absolute occupancy fidelity alongside changes in the builder's spatial anchor. Direct visual access improves translation-aligned geometry on a stratified subset, yet absolute grounding remains poor, indicating that current reconstruction failures extend beyond textualization alone. Automated provenance found no GT or default insertion, but unsupported fields and the absence of human semantic-equivalence assessment limit stronger equivalence claims.

## Threats to validity

Closed-model drift; regenerated Gemini cohort; a single primary fixed builder; stochastic generation; prompt dependence; automated content-preservation audit only; no human semantic-equivalence evaluation; local coordinate conventions; offline voxel execution; absence of Minecraft physics, navigation, and reachability; no human reconstruction baseline; modality-specific direct prompt; and subset-only direct and stochastic analyses.
