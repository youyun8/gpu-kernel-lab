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
  float avgMs;
  double gbPerSec;
  double gflopPerSec;
};

// Time a launch closure with warmup + steady-state GPU-event timing.
// The closure should enqueue exactly one kernel launch per call.
inline BenchResult benchmarkKernel(const std::function<void()>& launch, size_t bytesMoved,
                                   double flopCount, int warmup = kDefaultWarmup,
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

  float totalMs = 0.0f;
  GPU_CHECK(gpuEventElapsedTime(&totalMs, start, stop));
  GPU_CHECK(gpuEventDestroy(start));
  GPU_CHECK(gpuEventDestroy(stop));

  const float avgMs = totalMs / static_cast<float>(iters);
  const double seconds = static_cast<double>(avgMs) / 1.0e3;
  BenchResult result{};
  result.avgMs = avgMs;
  result.gbPerSec = static_cast<double>(bytesMoved) / seconds / 1.0e9;
  result.gflopPerSec = flopCount / seconds / 1.0e9;
  return result;
}

// Relative-error check of a device result against a host reference.
inline bool verifyClose(const std::vector<float>& got, const std::vector<float>& want,
                        float tol = 1.0e-3f) {
  if (got.size() != want.size()) {
    std::fprintf(stderr, "size mismatch: got %zu want %zu\n", got.size(), want.size());
    return false;
  }
  double maxRel = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const float denom = std::max(1.0e-6f, std::fabs(want[i]));
    const double rel = std::fabs(got[i] - want[i]) / denom;
    maxRel = std::max(maxRel, rel);
  }
  if (maxRel > tol) {
    std::fprintf(stderr, "correctness FAILED: max relative error %.3e > tol %.3e\n", maxRel, tol);
    return false;
  }
  std::printf("correctness OK (max relative error %.3e)\n", maxRel);
  return true;
}

// Print a one-line report including percentage of theoretical peak.
inline void report(const char* name, const BenchResult& r, double peakGbPerSec,
                   double peakGflopPerSec) {
  const double bwPct = peakGbPerSec > 0.0 ? 100.0 * r.gbPerSec / peakGbPerSec : 0.0;
  const double flopPct = peakGflopPerSec > 0.0 ? 100.0 * r.gflopPerSec / peakGflopPerSec : 0.0;
  std::printf("%-28s %8.3f ms | %8.1f GB/s (%.1f%% peak) | %9.1f GFLOP/s (%.1f%% peak)\n", name,
              r.avgMs, r.gbPerSec, bwPct, r.gflopPerSec, flopPct);
}

}  // namespace gklab
