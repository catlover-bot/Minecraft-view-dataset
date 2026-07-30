# Structured Absolute-IoU decomposition

Across 600 pairs, Structured−Free-form Absolute Occupancy IoU was -0.0095 (95% CI [-0.0134, -0.0057]), bbox dimension MAE was -0.2228, centroid distance was +1.067 blocks, and diagnostic position-grounding loss was +0.0138. Residual intrinsic geometry error did not increase robustly.

Predicted bbox minima shifted from (-4.50, -0.07, -5.01) to (-5.68, -0.12, -6.15). Mean best shifts changed from (6.42, 3.95, 5.09) to (7.31, 4.03, 6.04). The y-axis change was small relative to x/z, so ground-level interpretation was not the dominant axis-level change.

Mean voxel count changed from 3379.5 to 3366.5, bbox volume from 6450.8 to 6621.8, and occupancy density from 0.5957 to 0.5918. Add cost decreased by -0.0118; delete and replace costs changed by +0.0099 and +0.0100.

In the building-cluster-robust regression, increased centroid distance was the strongest measured correlate of the Absolute-IoU decline. Position-grounding loss is a diagnostic difference (Aligned minus Absolute IoU), not a causal decomposition.
