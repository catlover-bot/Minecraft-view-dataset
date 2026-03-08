#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/relocate_i2t2b_outputs.sh \
    --dataset_root datasets/buildings_100_v1 \
    [--out_root outputs/i2t2b]

Moves experiment outputs out of dataset_root:
  - building_xxx/{description*,rebuild_plan*,rebuild_world*,logs}
  - dataset_root/{metrics,logs}
to:
  out_root/<dataset_name>/...
USAGE
}

DATASET_ROOT=""
OUT_ROOT="outputs/i2t2b"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset_root)
      DATASET_ROOT="${2:-}"
      shift 2
      ;;
    --out_root)
      OUT_ROOT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${DATASET_ROOT}" ]]; then
  echo "--dataset_root is required" >&2
  usage
  exit 1
fi

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "dataset_root not found: ${DATASET_ROOT}" >&2
  exit 1
fi

DATASET_NAME="$(basename "${DATASET_ROOT}")"
DST_ROOT="${OUT_ROOT}/${DATASET_NAME}"
mkdir -p "${DST_ROOT}"

if [[ -d "${DATASET_ROOT}/metrics" ]]; then
  mv "${DATASET_ROOT}/metrics" "${DST_ROOT}/"
fi
if [[ -d "${DATASET_ROOT}/logs" ]]; then
  mv "${DATASET_ROOT}/logs" "${DST_ROOT}/"
fi

find "${DATASET_ROOT}" -maxdepth 1 -type d -name 'building_*' | while IFS= read -r b; do
  bn="$(basename "${b}")"
  outb="${DST_ROOT}/${bn}"
  mkdir -p "${outb}"
  find "${b}" -maxdepth 1 -type d \( -name 'logs' -o -name 'description*' -o -name 'rebuild_plan*' -o -name 'rebuild_world*' \) \
    | while IFS= read -r d; do
        mv "${d}" "${outb}/"
      done
done

echo "Relocated outputs:"
echo "  from: ${DATASET_ROOT}"
echo "  to:   ${DST_ROOT}"

