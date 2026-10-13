# Actual voxel projections

The two PNGs render actual GT and the fourteen successful saved voxel artifacts using `offline_evaluation/goal005/projections.py` and installed Pillow. Two failed outputs appear as MISSING. No Minecraft/Malmo, generation, GT repair, alignment or synthetic reconstruction was used.

Top is X/Z, front is X/Y, and side is Z/Y. The greatest-depth occupied cell determines the visible material. Absolute coordinate extents are shared across all rows of a scene. Colors are deterministic hashes of exact material IDs and the legend names those IDs; projection overlap can hide differences, so use the complete voxel confusion tables for quantitative conclusions.

Regenerate only into a new directory with the module's `--plan` and `--destination` options. `projections.json` records all sixteen cases and hashes the fourteen actual voxel inputs. Rendering does not recompute metrics or send requests.
