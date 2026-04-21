#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DATASET="buildings_100_v4"
SRC="datasets/${DATASET}"
DST="outputs/i2t2b/${DATASET}"
GEMINI_TAG="gemini_gemini_3_1_pro_preview"

RUN_PATTERN="tools/run_i2t2b_experiment.py --dataset_root datasets/${DATASET} --provider gemini"

echo "[gemini-v4-post] waiting running pipeline: ${RUN_PATTERN}"
while ps -ef | grep -F "${RUN_PATTERN}" | grep -v grep >/dev/null 2>&1; do
  sleep 20
done
echo "[gemini-v4-post] main pipeline finished. start postprocess."

mkdir -p "${DST}/metrics/description" "${DST}/metrics/rebuild" "${DST}/metrics/repair"

if [[ -d "${SRC}/metrics/description" ]]; then
  find "${SRC}/metrics/description" -maxdepth 1 -type f -name '*gemini*' -print0 \
    | while IFS= read -r -d '' f; do mv "${f}" "${DST}/metrics/description/"; done
fi
if [[ -d "${SRC}/metrics/rebuild" ]]; then
  find "${SRC}/metrics/rebuild" -maxdepth 1 -type f -name '*gemini*' -print0 \
    | while IFS= read -r -d '' f; do mv "${f}" "${DST}/metrics/rebuild/"; done
fi

find "${SRC}" -maxdepth 1 -type d -name 'building_*' -print0 | while IFS= read -r -d '' b; do
  bn="$(basename "${b}")"
  outb="${DST}/${bn}"
  mkdir -p "${outb}"
  find "${b}" -maxdepth 1 -type d -name '*gemini*' -print0 \
    | while IFS= read -r -d '' d; do
        mv "${d}" "${outb}/"
      done
done

python3 tools/evaluate_rebuild_metrics.py \
  --gt_root "datasets/${DATASET}" \
  --pred_root "${DST}" \
  --pred_subdir "rebuild_world_schema_material_v5_repair_${GEMINI_TAG}_common_v8_struct_self_refine_no_gt_tuned" \
  --pred_source rebuild_world \
  --building_pattern 'building_*' \
  --out "${DST}/metrics/rebuild/schema_v5_repair_${GEMINI_TAG}_self_refine_common_v8_struct_full.json"

python3 tools/evaluate_repair_effort.py \
  --gt_root "datasets/${DATASET}" \
  --pred_root "${DST}" \
  --pred_subdir "rebuild_world_schema_material_v5_repair_${GEMINI_TAG}_common_v8_struct_self_refine_no_gt_tuned" \
  --building_pattern 'building_*' \
  --max_shift_xy 48 \
  --max_shift_y 8 \
  --top_shift_candidates 24 \
  --out "${DST}/metrics/repair/gemini_main_direct_common_v8_struct.json"

python3 tools/build_structured_intermediate.py \
  --dataset_root "${DST}" \
  --description_subdir "description_${GEMINI_TAG}" \
  --out_subdir structured_intermediate_structured_ir_gemini_main_orig_20260419 \
  --building_pattern 'building_*'

python3 tools/generate_plan_from_intermediate.py \
  --dataset_root "${DST}" \
  --intermediate_subdir structured_intermediate_structured_ir_gemini_main_orig_20260419 \
  --out_subdir rebuild_plan_structured_ir_gemini_main_orig_20260419 \
  --building_pattern 'building_*'

python3 tools/render_rebuild_from_plan.py \
  --dataset_root "${DST}" \
  --plan_subdir rebuild_plan_structured_ir_gemini_main_orig_20260419 \
  --out_subdir rebuild_world_structured_ir_gemini_main_orig_20260419 \
  --building_pattern 'building_*'

python3 tools/evaluate_rebuild_metrics.py \
  --gt_root "datasets/${DATASET}" \
  --pred_root "${DST}" \
  --pred_subdir rebuild_world_structured_ir_gemini_main_orig_20260419 \
  --pred_source rebuild_world \
  --building_pattern 'building_*' \
  --out "${DST}/metrics/rebuild/structured_ir_gemini_main_orig_20260419.json"

python3 tools/evaluate_repair_effort.py \
  --gt_root "datasets/${DATASET}" \
  --pred_root "${DST}" \
  --pred_subdir rebuild_world_structured_ir_gemini_main_orig_20260419 \
  --building_pattern 'building_*' \
  --max_shift_xy 48 \
  --max_shift_y 8 \
  --top_shift_candidates 24 \
  --out "${DST}/metrics/repair/structured_ir_gemini_main_orig_20260419.json"

echo "[gemini-v4-post] done."
