#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_FILE="logs/wait_and_push_rebuild_captures.log"
mkdir -p logs

check_counts() {
  python3 - <<'PY'
import glob
import os

targets = [
    ("claude",
     "capture_rebuild_world_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned",
     "rebuild_world_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned"),
    ("gpt",
     "capture_rebuild_world_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned",
     "rebuild_world_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned"),
]
bases = ["outputs/i2t2b/buildings_100_v1", "outputs/i2t2b/buildings_100_v4"]

lines = []
all_ok = True
for name, cap, rw in targets:
    need = 0
    done = 0
    for base in bases:
        for b in glob.glob(os.path.join(base, "building_*")):
            if not os.path.isdir(os.path.join(b, rw)):
                continue
            need += 1
            meta = os.path.join(b, cap, "meta.json")
            img_dir = os.path.join(b, cap, "images")
            pngs = glob.glob(os.path.join(img_dir, "*.png")) if os.path.isdir(img_dir) else []
            if os.path.isfile(meta) and len(pngs) >= 8:
                done += 1
    lines.append(f"{name}:{done}/{need}")
    if done < need:
        all_ok = False

print(" ".join(lines))
raise SystemExit(0 if all_ok else 1)
PY
}

echo "[wait_and_push] started at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"

until check_counts >>"${LOG_FILE}" 2>&1; do
  echo "[wait_and_push] waiting... $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"
  sleep 120
done

echo "[wait_and_push] all captures completed. committing..." | tee -a "${LOG_FILE}"

git add \
  scripts/start_malmo_client_mac.sh \
  scripts/run_rebuild_world_capture.sh \
  scripts/run_batch_rebuild_world_capture_claude_tuned.sh \
  scripts/run_batch_rebuild_world_capture_tuned.sh \
  scripts/wait_and_push_rebuild_captures.sh \
  tools/capture_rebuild_world.py \
  outputs/i2t2b/buildings_100_v1/building_*/capture_rebuild_world_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned \
  outputs/i2t2b/buildings_100_v1/building_*/capture_rebuild_world_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned \
  outputs/i2t2b/buildings_100_v4/building_*/capture_rebuild_world_schema_material_v5_repair_anthropic_claude_haiku_4_5_20251001_self_refine_no_gt_tuned \
  outputs/i2t2b/buildings_100_v4/building_*/capture_rebuild_world_schema_material_v5_repair_openai_gpt_5_mini_self_refine_no_gt_tuned

if git diff --cached --quiet; then
  echo "[wait_and_push] no staged changes. skipping commit." | tee -a "${LOG_FILE}"
else
  git commit -m "Add full rebuild-world captures for Claude and GPT" | tee -a "${LOG_FILE}"
fi

git push | tee -a "${LOG_FILE}"
echo "[wait_and_push] push completed at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_FILE}"

