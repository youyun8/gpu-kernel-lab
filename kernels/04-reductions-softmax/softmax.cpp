// softmax.cpp
//
// Row-wise softmax comparing a three-pass approach with a single-pass online
// softmax (chapter 10). Both subtract the row max for numerical stability and
// are validated against a CPU reference, including a large-value overflow test.
#include <cmath>
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;

// Three-pass: find max, sum exp, normalize. One block per row.
__global__ void softmaxThreePass(const float* in, float* out, int rows, int cols) {
  __shared__ float reduce[kBlockSize];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) return;
  const float* x = in + static_cast<size_t>(row) * cols;
  float* y = out + static_cast<size_t>(row) * cols;

  float localMax = -INFINITY;
  for (int c = tid; c < cols; c += blockDim.x) localMax = fmaxf(localMax, x[c]);
  reduce[tid] = localMax;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
    __syncthreads();
  }
  const float rowMax = reduce[0];
  __syncthreads();

  float localSum = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) localSum += expf(x[c] - rowMax);
  reduce[tid] = localSum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) reduce[tid] += reduce[tid + s];
    __syncthreads();
  }
  const float rowSum = reduce[0];
  __syncthreads();

  for (int c = tid; c < cols; c += blockDim.x) y[c] = expf(x[c] - rowMax) / rowSum;
}

// Online softmax: one pass maintaining running max and running denominator with
// rescale, then a normalization pass. One block per row.
__global__ void softmaxOnline(const float* in, float* out, int rows, int cols) {
  __shared__ float maxReduce[kBlockSize];
  __shared__ float sumReduce[kBlockSize];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) return;
  const float* x = in + static_cast<size_t>(row) * cols;
  float* y = out + static_cast<size_t>(row) * cols;

  float m = -INFINITY;  // running max
  float d = 0.0f;       // running denominator
  for (int c = tid; c < cols; c += blockDim.x) {
    const float v = x[c];
    const float mNew = fmaxf(m, v);
    d = d * expf(m - mNew) + expf(v - mNew);
    m = mNew;
  }
  maxReduce[tid] = m;
  sumReduce[tid] = d;
  __syncthreads();

  // Combine per-thread (m, d) pairs pairwise in shared memory.
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      const float m1 = maxReduce[tid];
      const float d1 = sumReduce[tid];
      const float m2 = maxReduce[tid + s];
      const float d2 = sumReduce[tid + s];
      const float mNew = fmaxf(m1, m2);
      maxReduce[tid] = mNew;
      sumReduce[tid] = d1 * expf(m1 - mNew) + d2 * expf(m2 - mNew);
    }
    __syncthreads();
  }
  const float rowMax = maxReduce[0];
  const float rowSum = sumReduce[0];
  __syncthreads();

  for (int c = tid; c < cols; c += blockDim.x) y[c] = expf(x[c] - rowMax) / rowSum;
}

void cpuSoftmax(const std::vector<float>& in, std::vector<float>& out, int rows, int cols) {
  for (int r = 0; r < rows; ++r) {
    const float* x = in.data() + static_cast<size_t>(r) * cols;
    float* y = out.data() + static_cast<size_t>(r) * cols;
    float m = -INFINITY;
    for (int c = 0; c < cols; ++c) m = std::fmax(m, x[c]);
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) sum += std::exp(x[c] - m);
    for (int c = 0; c < cols; ++c) y[c] = static_cast<float>(std::exp(x[c] - m) / sum);
  }
}

}  // namespace

int main() {
  constexpr int kRows = 4096;
  constexpr int kCols = 1024;
  const size_t count = static_cast<size_t>(kRows) * kCols;
  const size_t bytes = count * sizeof(float);

  std::vector<float> host(count);
  for (size_t i = 0; i < count; ++i) {
    // Include large values to exercise numerical stability.
    host[i] = static_cast<float>((i % 200) - 100) * (i % 997 == 0 ? 50.0f : 0.1f);
  }
  std::vector<float> reference(count);
  cpuSoftmax(host, reference, kRows, kCols);
  std::vector<float> result(count);

  float* devIn = nullptr;
  float* devOut = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devIn), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devOut), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(devIn, host.data(), bytes));

  constexpr double kPeakGbPerSec = 1555.0;
  const size_t bytesMoved = 2 * bytes;

  auto check = [&](const char* name, const std::function<void()>& launch) -> bool {
    launch();
    GPU_CHECK(gpuDeviceSynchronize());
    GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devOut, bytes));
    if (!gklab::verifyClose(result, reference, 1.0e-3f)) {
      std::fprintf(stderr, "%s FAILED\n", name);
      return false;
    }
    gklab::report(name, gklab::benchmarkKernel(launch, bytesMoved, 0.0), kPeakGbPerSec, 0.0);
    return true;
  };

  if (!check("softmax_three_pass", [&]() { GPU_LAUNCH(softmaxThreePass, kRows, kBlockSize, 0, devIn, devOut, kRows, kCols); }))
    return EXIT_FAILURE;
  if (!check("softmax_online", [&]() { GPU_LAUNCH(softmaxOnline, kRows, kBlockSize, 0, devIn, devOut, kRows, kCols); }))
    return EXIT_FAILURE;

  GPU_CHECK(gpuFree(devIn));
  GPU_CHECK(gpuFree(devOut));
  return EXIT_SUCCESS;
}
