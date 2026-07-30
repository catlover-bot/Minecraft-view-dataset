from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from .common import load
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/arr_followup_v1.yaml");a=p.parse_args();c=load(a.config);phase=pd.read_csv(c["_out"]/"analysis"/"structured_absolute_iou"/"aggregate.csv");reg=pd.read_csv(c["_out"]/"analysis"/"structured_absolute_iou"/"regression_results.csv");comp=pd.read_csv(c["_out"]/"analysis"/"completeness_repair"/"adjusted_models.csv");direct=pd.read_csv(c["_out"]/"analysis"/"direct_image_to_build"/"aggregate_metrics.csv");tests=pd.read_csv(c["_out"]/"analysis"/"direct_image_to_build"/"paired_tests.csv");audit=(c["_out"]/"analysis"/"ir_content_preservation"/"audit_summary.md").read_text();st=(c["_out"]/"stochastic_repeat"/"summary.md").read_text() if (c["_out"]/"stochastic_repeat"/"summary.md").is_file() else "Stochastic repeat unavailable."
 def val(metric):return phase[(phase.model=="pooled")&(phase.metric==metric)].iloc[0]
 ai=val("delta_absolute_iou");cent=val("delta_centroid_distance");pos=val("delta_position_grounding_loss");bbox=val("delta_bbox_dimension_mae");ce=comp[comp.term=="completeness_score"].iloc[0]
 claims=f"""# Final ARR claims

## Recommended RQs

1. Which properties of generated descriptions predict downstream reconstruction performance when builder capability is controlled?
2. How does description-only structural canonicalization, with automated provenance auditing and documented unsupported fields, affect geometric, material, and positional reconstruction fidelity?
3. Where does transfer from attribute recognition to discrete spatial grounding break down?
4. On a stratified 100-building subset, how does direct image-to-build generation compare with language-mediated reconstruction using the same model family?

Here, content-preserving is limited to provenance: generated solely from the corresponding free-form description, without access to images, ground truth, or inserted default values. The strengthened audit identified unsupported duplicated top-level fields in a subset, so perfect field or semantic preservation is not claimed and a strict-supported sensitivity analysis is reported.

## RQ answers

RQ1. Description Overall moderately predicts absolute and aligned geometry; dimension and material-description scores predict the corresponding explicit attributes more strongly than exact spatial grounding. These are associations under one controlled builder, not causal effects.

RQ2. Across 600 pairs, canonicalization reduced Absolute Occupancy IoU by {ai['mean']:.4f} (95% CI [{ai['ci95_low']:.4f}, {ai['ci95_high']:.4f}]) while reducing bbox dimension MAE by {-bbox['mean']:.4f} blocks. It did not yield a robust aligned-IoU improvement; increased centroid displacement and diagnostic position-grounding loss are consistent with the absolute-IoU decline.

RQ3. Attribute identity transfers better than absolute grounding. Position-grounding loss increased by {pos['mean']:.4f}, and centroid distance increased by {cent['mean']:.3f} blocks under canonicalization, while material selection remained much stronger than exact position-and-material agreement.

RQ4. Direct image-to-build used the same gpt-5-mini model family on a stratified n=100 subset. It achieved higher aligned geometry and lower bbox error but lower Absolute Occupancy IoU, showing that direct visual access did not solve coordinate anchoring; results do not support a uniformly superior direct pathway.

## Supported claims

- Canonicalization improved bbox dimensional fidelity but reduced absolute positional occupancy fidelity in n=600 paired observations; exact CIs and Holm-adjusted tests are in the parent and follow-up tables.
- The Absolute-IoU decline is associated most strongly with increased centroid displacement in the building-cluster-robust regression; it is not explained by an aligned-geometry decline.
- Completeness remained positively associated with the original GT-normalized repair cost after model, GT size, bbox volume, material count, and description length adjustment (coefficient {ce['estimate']:.3f}, 95% CI [{ce['ci95_low']:.3f}, {ce['ci95_high']:.3f}], p={ce['p']:.4g}, n=600).
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
"""
 (c["_out"]/"arr_claims_final.md").write_text(claims)
 per=pd.read_csv(c["_out"]/"analysis"/"structured_absolute_iou"/"per_scene.csv")
 axis={k:per[k].mean() for k in ["free_min_x","structured_min_x","free_min_y","structured_min_y","free_min_z","structured_min_z","free_alignment_shift_x","structured_alignment_shift_x","free_alignment_shift_y","structured_alignment_shift_y","free_alignment_shift_z","structured_alignment_shift_z","free_voxel_count","structured_voxel_count","free_bbox_volume","structured_bbox_volume","free_occupancy_density","structured_occupancy_density","free_normalized_add_cost","structured_normalized_add_cost","free_normalized_delete_cost","structured_normalized_delete_cost","free_normalized_replace_cost","structured_normalized_replace_cost"]}
 (c["_out"]/"analysis"/"structured_absolute_iou"/"summary.md").write_text(f"""# Structured Absolute-IoU decomposition

Across 600 pairs, Structured−Free-form Absolute Occupancy IoU was {ai['mean']:.4f} (95% CI [{ai['ci95_low']:.4f}, {ai['ci95_high']:.4f}]), bbox dimension MAE was {bbox['mean']:.4f}, centroid distance was {cent['mean']:+.3f} blocks, and diagnostic position-grounding loss was {pos['mean']:+.4f}. Residual intrinsic geometry error did not increase robustly.

Predicted bbox minima shifted from ({axis['free_min_x']:.2f}, {axis['free_min_y']:.2f}, {axis['free_min_z']:.2f}) to ({axis['structured_min_x']:.2f}, {axis['structured_min_y']:.2f}, {axis['structured_min_z']:.2f}). Mean best shifts changed from ({axis['free_alignment_shift_x']:.2f}, {axis['free_alignment_shift_y']:.2f}, {axis['free_alignment_shift_z']:.2f}) to ({axis['structured_alignment_shift_x']:.2f}, {axis['structured_alignment_shift_y']:.2f}, {axis['structured_alignment_shift_z']:.2f}). The y-axis change was small relative to x/z, so ground-level interpretation was not the dominant axis-level change.

Mean voxel count changed from {axis['free_voxel_count']:.1f} to {axis['structured_voxel_count']:.1f}, bbox volume from {axis['free_bbox_volume']:.1f} to {axis['structured_bbox_volume']:.1f}, and occupancy density from {axis['free_occupancy_density']:.4f} to {axis['structured_occupancy_density']:.4f}. Add cost decreased by {axis['structured_normalized_add_cost']-axis['free_normalized_add_cost']:+.4f}; delete and replace costs changed by {axis['structured_normalized_delete_cost']-axis['free_normalized_delete_cost']:+.4f} and {axis['structured_normalized_replace_cost']-axis['free_normalized_replace_cost']:+.4f}.

In the building-cluster-robust regression, increased centroid distance was the strongest measured correlate of the Absolute-IoU decline. Position-grounding loss is a diagnostic difference (Aligned minus Absolute IoU), not a causal decomposition.
""")
 cr=pd.read_csv(c["_out"]/"analysis"/"completeness_repair"/"correlations.csv");pool=cr[cr.model=="pooled"].set_index("outcome").spearman_rho
 ns=pd.read_csv(c["_out"]/"analysis"/"completeness_repair"/"normalization_sensitivity.csv");nsp=ns[ns.model=="pooled"].set_index("normalization").spearman_rho
 (c["_out"]/"analysis"/"completeness_repair"/"summary.md").write_text(f"""# Completeness and repair burden

Completeness correlated with action count (rho={pool['free_action_count']:.3f}), predicted voxel count (rho={pool['free_voxel_count']:.3f}), predicted/GT voxel ratio (rho={pool['free_voxel_count_ratio']:.3f}), delete cost (rho={pool['free_normalized_delete_cost']:.3f}), replace cost (rho={pool['free_normalized_replace_cost']:.3f}), and total repair cost (rho={pool['free_total_normalized_repair_cost']:.3f}); add cost was negatively associated (rho={pool['free_normalized_add_cost']:.3f}). OpenAI and Claude completeness scores were constant, while the within-model relationship was estimable for Gemini, so model composition and Gemini variation materially contribute.

After adjustment for description model, GT voxel count, bbox volume, material count, and description length with building-cluster-robust standard errors, the completeness coefficient remained {ce['estimate']:.3f} (95% CI [{ce['ci95_low']:.3f}, {ce['ci95_high']:.3f}], p={ce['p']:.4g}). This is associational.

Normalization sensitivity was substantial: rho={nsp['original']:.3f} under the original GT normalization, {nsp['max_norm']:.3f} under max(GT,pred), and {nsp['sum_norm']:.3f} under GT+pred normalization. Thus A (association with greater generation/delete burden), C (model/Gemini contribution), D (normalization contribution), and E (adjusted association) are supported; B (fully explained by building complexity) is not supported.
""")
 lim="""# Limitations

Although the structured representations were generated solely from their corresponding free-form descriptions and passed automated field-level provenance checks, we did not conduct a human semantic equivalence study. Subtle changes in ambiguity, emphasis, uncertainty, or spatial interpretation may therefore remain.

Our reconstructions are evaluated through a deterministic offline voxel executor rather than embodied agent execution in Minecraft. We do not model Minecraft physics, navigation, reachability, block-support constraints, interaction sequences, or execution failures caused by an agent's viewpoint and movement. Consequently, the results characterize discrete voxel reconstruction under the repository action semantics, not end-to-end Minecraft task completion.

The experiments use closed models whose behavior may drift despite fixed model identifiers, prompts, and logged response metadata. The Gemini description cohort was regenerated under a unified 2026 condition and may not be directly interchangeable with earlier closed-model outputs. Most controlled reconstruction results use a single OpenAI fixed builder, so representation effects may depend on that builder's capabilities and prompt interpretation.

The stochastic analysis uses a stratified subset rather than all 200 buildings. The Direct Image-to-Build analysis likewise uses a stratified 100-building subset and should not be generalized to the full benchmark without qualification. Direct and language-mediated conditions share the gpt-5-mini model identifier, validator, executor, and action schema, but require modality-specific prompts; thus, differences cannot be attributed solely to the presence or absence of textualization.

Finally, we did not evaluate reconstruction by human builders and make no claim that the observed effects generalize to human interpretation or construction.
"""
 out=c["_out"]/"paper_text";out.mkdir(parents=True,exist_ok=True);(out/"limitations.md").write_text(lim);tex=lim.replace("# Limitations","\\section{Limitations}").replace("\n\n","\\par\n").replace("_","\\_");(out/"limitations.tex").write_text(tex)
if __name__=="__main__":main()
