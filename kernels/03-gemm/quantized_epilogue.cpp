// quantized_epilogue.cpp
//
// Chapter 29 reference solution: a small GEMM that keeps fp32 accumulation,
// fuses bias, and requantizes to int8 in the epilogue with per-column scale.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

__global__ void gemmQuantizedEpilogue(const float* a, const float* b, const float* bias,
                                      const float* invScale, signed char* c, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;

  float acc = 0.0f;
  for (int p = 0; p < k; ++p) acc += a[row * k + p] * b[p * n + col];
  float y = acc + bias[col];
  float q = nearbyintf(y * invScale[col]);
  q = fminf(127.0f, fmaxf(-128.0f, q));
  c[row * n + col] = static_cast<signed char>(q);
}

signed char quantize(float value, float invScale) {
  float q = std::nearbyint(value * invScale);
  q = std::min(127.0f, std::max(-128.0f, q));
  return static_cast<signed char>(q);
}

void cpuReference(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& bias,
                  const std::vector<float>& invScale, std::vector<signed char>& c, int m, int n, int k) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int p = 0; p < k; ++p) acc += a[row * k + p] * b[p * n + col];
      c[row * n + col] = quantize(acc + bias[col], invScale[col]);
    }
  }
}

bool verifyInt8(const std::vector<signed char>& got, const std::vector<signed char>& want) {
  int mismatches = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      if (++mismatches <= 8) {
        std::fprintf(stderr, "mismatch at %zu: got %d want %d\n", i, static_cast<int>(got[i]),
                     static_cast<int>(want[i]));
      }
    }
  }
  if (mismatches != 0) {
    std::fprintf(stderr, "quantized epilogue FAILED: %d mismatches\n", mismatches);
    return false;
  }
  std::printf("quantized epilogue correctness OK\n");
  return true;
}

}  // namespace

int main() {
  constexpr int kM = 128;
  constexpr int kN = 128;
  constexpr int kK = 256;
  const size_t bytesA = static_cast<size_t>(kM) * kK * sizeof(float);
  const size_t bytesB = static_cast<size_t>(kK) * kN * sizeof(float);
  const size_t bytesBias = kN * sizeof(float);
  const size_t bytesScale = kN * sizeof(float);
  const size_t bytesC = static_cast<size_t>(kM) * kN * sizeof(signed char);

  std::vector<float> hostA(static_cast<size_t>(kM) * kK);
  std::vector<float> hostB(static_cast<size_t>(kK) * kN);
  std::vector<float> bias(kN);
  std::vector<float> invScale(kN);
  std::vector<signed char> reference(static_cast<size_t>(kM) * kN);
  std::vector<signed char> result(static_cast<size_t>(kM) * kN);

  for (size_t i = 0; i < hostA.size(); ++i) hostA[i] = static_cast<float>((i % 17) - 8) * 0.01f;
  for (size_t i = 0; i < hostB.size(); ++i) hostB[i] = static_cast<float>((i % 13) - 6) * 0.02f;
  for (int col = 0; col < kN; ++col) {
    bias[col] = static_cast<float>((col % 11) - 5) * 0.05f;
    invScale[col] = 8.0f + static_cast<float>(col % 7);
  }
  cpuReference(hostA, hostB, bias, invScale, reference, kM, kN, kK);

  float* devA = nullptr;
  float* devB = nullptr;
  float* devBias = nullptr;
  float* devInvScale = nullptr;
  signed char* devC = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devA), bytesA));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devB), bytesB));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devBias), bytesBias));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devInvScale), bytesScale));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devC), bytesC));
  GPU_CHECK(gpuMemcpyHostToDevice(devA, hostA.data(), bytesA));
  GPU_CHECK(gpuMemcpyHostToDevice(devB, hostB.data(), bytesB));
  GPU_CHECK(gpuMemcpyHostToDevice(devBias, bias.data(), bytesBias));
  GPU_CHECK(gpuMemcpyHostToDevice(devInvScale, invScale.data(), bytesScale));

  dim3 block(kBlockX, kBlockY);
  dim3 grid((kN + kBlockX - 1) / kBlockX, (kM + kBlockY - 1) / kBlockY);
  auto launch = [&]() {
    GPU_LAUNCH(gemmQuantizedEpilogue, grid, block, 0, devA, devB, devBias, devInvScale, devC, kM, kN, kK);
  };

  launch();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devC, bytesC));
  if (!verifyInt8(result, reference)) return EXIT_FAILURE;

  const size_t bytesMoved = bytesA + bytesB + bytesBias + bytesScale + bytesC;
  const double flops = 2.0 * kM * kN * kK;
  constexpr double kPeakGbPerSec = 1555.0;
  constexpr double kPeakGflopPerSec = 19500.0;
  gklab::report("gemm_quantized_epilogue", gklab::benchmarkKernel(launch, bytesMoved, flops),
                kPeakGbPerSec, kPeakGflopPerSec);

  GPU_CHECK(gpuFree(devA));
  GPU_CHECK(gpuFree(devB));
  GPU_CHECK(gpuFree(devBias));
  GPU_CHECK(gpuFree(devInvScale));
  GPU_CHECK(gpuFree(devC));
  return EXIT_SUCCESS;
}
