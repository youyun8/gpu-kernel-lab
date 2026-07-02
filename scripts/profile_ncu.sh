#!/usr/bin/env bash
# profile_ncu.sh — Nsight Compute recipes for the GEMM optimization series.
#
# Usage:
#   scripts/profile_ncu.sh [build_dir]
#
# Requires an NVIDIA GPU with Nsight Compute (ncu) installed. Collects the
# speed-of-light and memory workload sections for each GEMM step so you can
# watch the bottleneck migrate from memory-bound to compute-bound (chapter 16).
set -euo pipefail

BUILD_DIR="${1:-kernels/build}"
OUT_DIR="ncu_reports"
mkdir -p "${OUT_DIR}"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu (Nsight Compute) not found. Install the CUDA toolkit / Nsight Compute." >&2
  exit 1
fi

STEPS=(
  "03-gemm/sgemm_step0_naive"
  "03-gemm/sgemm_step1_shared"
  "03-gemm/sgemm_step2_register_tiling"
  "03-gemm/sgemm_step3_vectorized"
  "03-gemm/sgemm_step4_double_buffer"
)

for step in "${STEPS[@]}"; do
  exe="${BUILD_DIR}/${step}"
  name="$(basename "${step}")"
  if [[ -x "${exe}" ]]; then
    echo "Profiling ${name} ..."
    # --set full collects SOL, memory workload, scheduler, and occupancy sections.
    ncu --set full --launch-count 1 --export "${OUT_DIR}/${name}" --force-overwrite "${exe}" || true
  else
    echo "skip ${exe} (not built)"
  fi
done

echo "Reports written to ${OUT_DIR}/. Open with: ncu-ui ${OUT_DIR}/<name>.ncu-rep"
