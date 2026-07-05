// benchmark.h
//
// Timing, validation, and reporting helpers shared by every kernel example.
// The harness performs warmup iterations, times steady-state iterations with
// GPU events, and reports achieved bandwidth (GB/s), throughput (GFLOP/s), and
// percentage of a user-supplied theoretical peak.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <vector>

#include "gpu_portability.h"

namespace gklab {

constexpr int kDefaultWarmup = 10;
constexpr int kDefaultIters = 100;

struct BenchResult {
  float avg_ms;
  double gb_per_sec;
  double gflop_per_sec;
};

// Time a launch closure with warmup + steady-state GPU-event timing.
// The closure should enqueue exactly one kernel launch per call.
inline BenchResult benchmarkKernel(const std::function<void()>& launch, size_t bytes_moved,
                                   double flop_count, int warmup = kDefaultWarmup,
                                   int iters = kDefaultIters) {
  for (int i = 0; i < warmup; ++i) launch();
  GPU_CHECK(gpuDeviceSynchronize());

  GpuEvent start;
  GpuEvent stop;
  GPU_CHECK(gpuEventCreate(&start));
  GPU_CHECK(gpuEventCreate(&stop));

  GPU_CHECK(gpuEventRecord(start));
  for (int i = 0; i < iters; ++i) launch();
  GPU_CHECK(gpuEventRecord(stop));
  GPU_CHECK(gpuEventSynchronize(stop));

  float total_ms = 0.0f;
  GPU_CHECK(gpuEventElapsedTime(&total_ms, start, stop));
  GPU_CHECK(gpuEventDestroy(start));
  GPU_CHECK(gpuEventDestroy(stop));

  const float avg_ms = total_ms / static_cast<float>(iters);
  const double seconds = static_cast<double>(avg_ms) / 1.0e3;
  BenchResult result{};
  result.avg_ms = avg_ms;
  result.gb_per_sec = static_cast<double>(bytes_moved) / seconds / 1.0e9;
  result.gflop_per_sec = flop_count / seconds / 1.0e9;
  return result;
}

// Relative-error check of a device result against a host reference.
inline bool verifyClose(const std::vector<float>& got, const std::vector<float>& want,
                        float tol = 1.0e-3f) {
  if (got.size() != want.size()) {
    std::fprintf(stderr, "size mismatch: got %zu want %zu\n", got.size(), want.size());
    return false;
  }
  double max_rel = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const float denom = std::max(1.0e-6f, std::fabs(want[i]));
    const double rel = std::fabs(got[i] - want[i]) / denom;
    max_rel = std::max(max_rel, rel);
  }
  if (max_rel > tol) {
    std::fprintf(stderr, "correctness FAILED: max relative error %.3e > tol %.3e\n", max_rel, tol);
    return false;
  }
  std::printf("correctness OK (max relative error %.3e)\n", max_rel);
  return true;
}

// Print a one-line report including percentage of theoretical peak.
inline void report(const char* name, const BenchResult& r, double peak_gb_per_sec,
                   double peak_gflop_per_sec) {
  const double bw_pct = peak_gb_per_sec > 0.0 ? 100.0 * r.gb_per_sec / peak_gb_per_sec : 0.0;
  const double flop_pct = peak_gflop_per_sec > 0.0 ? 100.0 * r.gflop_per_sec / peak_gflop_per_sec : 0.0;
  std::printf("%-28s %8.3f ms | %8.1f GB/s (%.1f%% peak) | %9.1f GFLOP/s (%.1f%% peak)\n", name,
              r.avg_ms, r.gb_per_sec, bw_pct, r.gflop_per_sec, flop_pct);
}

}  // namespace gklab
