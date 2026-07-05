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
                                      const float* inv_scale, signed char* c, int m, int n, int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) return;

  float acc = 0.0f;
  for (int p = 0; p < k; ++p) acc += a[row * k + p] * b[p * n + col];
  float y = acc + bias[col];
  float q = nearbyintf(y * inv_scale[col]);
  q = fminf(127.0f, fmaxf(-128.0f, q));
  c[row * n + col] = static_cast<signed char>(q);
}

signed char quantize(float value, float inv_scale) {
  float q = std::nearbyint(value * inv_scale);
  q = std::min(127.0f, std::max(-128.0f, q));
  return static_cast<signed char>(q);
}

void cpuReference(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& bias,
                  const std::vector<float>& inv_scale, std::vector<signed char>& c, int m, int n, int k) {
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int p = 0; p < k; ++p) acc += a[row * k + p] * b[p * n + col];
      c[row * n + col] = quantize(acc + bias[col], inv_scale[col]);
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
  constexpr int kSizeM = 128;
  constexpr int kSizeN = 128;
  constexpr int kSizeK = 256;
  const size_t bytes_a = static_cast<size_t>(kSizeM) * kSizeK * sizeof(float);
  const size_t bytes_b = static_cast<size_t>(kSizeK) * kSizeN * sizeof(float);
  const size_t bytes_bias = kSizeN * sizeof(float);
  const size_t bytes_scale = kSizeN * sizeof(float);
  const size_t bytes_c = static_cast<size_t>(kSizeM) * kSizeN * sizeof(signed char);

  std::vector<float> host_a(static_cast<size_t>(kSizeM) * kSizeK);
  std::vector<float> host_b(static_cast<size_t>(kSizeK) * kSizeN);
  std::vector<float> bias(kSizeN);
  std::vector<float> inv_scale(kSizeN);
  std::vector<signed char> reference(static_cast<size_t>(kSizeM) * kSizeN);
  std::vector<signed char> result(static_cast<size_t>(kSizeM) * kSizeN);

  for (size_t i = 0; i < host_a.size(); ++i) host_a[i] = static_cast<float>((i % 17) - 8) * 0.01f;
  for (size_t i = 0; i < host_b.size(); ++i) host_b[i] = static_cast<float>((i % 13) - 6) * 0.02f;
  for (int col = 0; col < kSizeN; ++col) {
    bias[col] = static_cast<float>((col % 11) - 5) * 0.05f;
    inv_scale[col] = 8.0f + static_cast<float>(col % 7);
  }
  cpuReference(host_a, host_b, bias, inv_scale, reference, kSizeM, kSizeN, kSizeK);

  float* dev_a = nullptr;
  float* dev_b = nullptr;
  float* dev_bias = nullptr;
  float* dev_inv_scale = nullptr;
  signed char* dev_c = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_a), bytes_a));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_b), bytes_b));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_bias), bytes_bias));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_inv_scale), bytes_scale));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_c), bytes_c));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_a, host_a.data(), bytes_a));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_b, host_b.data(), bytes_b));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_bias, bias.data(), bytes_bias));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_inv_scale, inv_scale.data(), bytes_scale));

  dim3 block(kBlockX, kBlockY);
  dim3 grid((kSizeN + kBlockX - 1) / kBlockX, (kSizeM + kBlockY - 1) / kBlockY);
  auto launch = [&]() {
    GPU_LAUNCH(gemmQuantizedEpilogue, grid, block, 0, dev_a, dev_b, dev_bias, dev_inv_scale, dev_c, kSizeM, kSizeN, kSizeK);
  };

  launch();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_c, bytes_c));
  if (!verifyInt8(result, reference)) return EXIT_FAILURE;

  const size_t bytes_moved = bytes_a + bytes_b + bytes_bias + bytes_scale + bytes_c;
  const double flops = 2.0 * kSizeM * kSizeN * kSizeK;
  constexpr double kPeakGbPerSec = 1555.0;
  constexpr double kPeakGflopPerSec = 19500.0;
  gklab::report("gemm_quantized_epilogue", gklab::benchmarkKernel(launch, bytes_moved, flops),
                kPeakGbPerSec, kPeakGflopPerSec);

  GPU_CHECK(gpuFree(dev_a));
  GPU_CHECK(gpuFree(dev_b));
  GPU_CHECK(gpuFree(dev_bias));
  GPU_CHECK(gpuFree(dev_inv_scale));
  GPU_CHECK(gpuFree(dev_c));
  return EXIT_SUCCESS;
}
