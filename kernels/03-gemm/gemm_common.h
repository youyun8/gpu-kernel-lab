// gemm_common.h
//
// Shared setup, CPU reference, and benchmark driver for the SGEMM optimization
// series (chapter 12). Each step file defines its own kernel and a launch
// closure, then calls runGemm() to validate and benchmark it.
#pragma once

#include <cstdio>
#include <functional>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace gklab {

// Row-major C = A (m x k) * B (k x n). Small size keeps the CPU reference fast.
inline void cpuGemm(const std::vector<float>& a, const std::vector<float>& b,
                    std::vector<float>& c, int m, int n, int k) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int p = 0; p < k; ++p) acc += a[row * k + p] * b[p * n + col];
      c[row * n + col] = acc;
    }
  }
}

struct GemmBuffers {
  float* a = nullptr;
  float* b = nullptr;
  float* c = nullptr;
};

// Runs the provided launch closure: correctness check vs. CPU reference, then
// benchmark reporting GFLOP/s and % of an illustrative peak.
inline int runGemm(const char* name, int m, int n, int k,
                   const std::function<void(const GemmBuffers&)>& launch) {
  const size_t bytesA = static_cast<size_t>(m) * k * sizeof(float);
  const size_t bytesB = static_cast<size_t>(k) * n * sizeof(float);
  const size_t bytesC = static_cast<size_t>(m) * n * sizeof(float);

  std::vector<float> hostA(static_cast<size_t>(m) * k);
  std::vector<float> hostB(static_cast<size_t>(k) * n);
  std::vector<float> hostC(static_cast<size_t>(m) * n);
  std::vector<float> reference(static_cast<size_t>(m) * n);
  for (size_t i = 0; i < hostA.size(); ++i) hostA[i] = static_cast<float>((i % 13) - 6) * 0.1f;
  for (size_t i = 0; i < hostB.size(); ++i) hostB[i] = static_cast<float>((i % 7) - 3) * 0.2f;
  cpuGemm(hostA, hostB, reference, m, n, k);

  GemmBuffers buf;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&buf.a), bytesA));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&buf.b), bytesB));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&buf.c), bytesC));
  GPU_CHECK(gpuMemcpyHostToDevice(buf.a, hostA.data(), bytesA));
  GPU_CHECK(gpuMemcpyHostToDevice(buf.b, hostB.data(), bytesB));

  launch(buf);
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(hostC.data(), buf.c, bytesC));
  const bool ok = verifyClose(hostC, reference, 1.0e-2f);

  const double flops = 2.0 * m * n * k;
  const size_t bytesMoved = bytesA + bytesB + bytesC;
  constexpr double kPeakGbPerSec = 1555.0;
  constexpr double kPeakGflopPerSec = 19500.0;
  report(name, benchmarkKernel([&]() { launch(buf); }, bytesMoved, flops), kPeakGbPerSec,
         kPeakGflopPerSec);

  GPU_CHECK(gpuFree(buf.a));
  GPU_CHECK(gpuFree(buf.b));
  GPU_CHECK(gpuFree(buf.c));
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace gklab
