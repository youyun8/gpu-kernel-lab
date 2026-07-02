#!/usr/bin/env bash
# profile_rocprof.sh — rocprofv3 / omniperf recipes for the GEMM series.
#
# Usage:
#   scripts/profile_rocprof.sh [build_dir]
#
# Requires an AMD GPU with ROCm. Uses rocprofv3 for stats and, if available,
# omniperf for the roofline / memory-chart analysis (chapter 16).
set -euo pipefail

BUILD_DIR="${1:-kernels/build}"
OUT_DIR="rocprof_reports"
mkdir -p "${OUT_DIR}"

STEPS=(
  "03-gemm/sgemm_step0_naive"
  "03-gemm/sgemm_step1_shared"
  "03-gemm/sgemm_step2_register_tiling"
  "03-gemm/sgemm_step3_vectorized"
  "03-gemm/sgemm_step4_double_buffer"
)

has_rocprofv3=0
has_omniperf=0
command -v rocprofv3 >/dev/null 2>&1 && has_rocprofv3=1
command -v omniperf  >/dev/null 2>&1 && has_omniperf=1

if [[ "${has_rocprofv3}" -eq 0 && "${has_omniperf}" -eq 0 ]]; then
  echo "Neither rocprofv3 nor omniperf found. Install ROCm profiling tools." >&2
  exit 1
fi

for step in "${STEPS[@]}"; do
  exe="${BUILD_DIR}/${step}"
  name="$(basename "${step}")"
  if [[ ! -x "${exe}" ]]; then
    echo "skip ${exe} (not built)"
    continue
  fi
  if [[ "${has_rocprofv3}" -eq 1 ]]; then
    echo "rocprofv3 stats for ${name} ..."
    rocprofv3 --stats --output-directory "${OUT_DIR}/${name}" -- "${exe}" || true
  fi
  if [[ "${has_omniperf}" -eq 1 ]]; then
    echo "omniperf profile for ${name} ..."
    omniperf profile -n "${name}" -- "${exe}" || true
  fi
done

echo "Reports written to ${OUT_DIR}/. Analyze omniperf with: omniperf analyze -p workloads/<name>/..."
