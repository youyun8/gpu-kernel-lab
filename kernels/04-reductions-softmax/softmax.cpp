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

  float local_max = -INFINITY;
  for (int c = tid; c < cols; c += blockDim.x) local_max = fmaxf(local_max, x[c]);
  reduce[tid] = local_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) reduce[tid] = fmaxf(reduce[tid], reduce[tid + s]);
    __syncthreads();
  }
  const float row_max = reduce[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) local_sum += expf(x[c] - row_max);
  reduce[tid] = local_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) reduce[tid] += reduce[tid + s];
    __syncthreads();
  }
  const float row_sum = reduce[0];
  __syncthreads();

  for (int c = tid; c < cols; c += blockDim.x) y[c] = expf(x[c] - row_max) / row_sum;
}

// Online softmax: one pass maintaining running max and running denominator with
// rescale, then a normalization pass. One block per row.
__global__ void softmaxOnline(const float* in, float* out, int rows, int cols) {
  __shared__ float max_reduce[kBlockSize];
  __shared__ float sum_reduce[kBlockSize];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  if (row >= rows) return;
  const float* x = in + static_cast<size_t>(row) * cols;
  float* y = out + static_cast<size_t>(row) * cols;

  float m = -INFINITY;  // running max
  float d = 0.0f;       // running denominator
  for (int c = tid; c < cols; c += blockDim.x) {
    const float v = x[c];
    const float m_new = fmaxf(m, v);
    d = d * expf(m - m_new) + expf(v - m_new);
    m = m_new;
  }
  max_reduce[tid] = m;
  sum_reduce[tid] = d;
  __syncthreads();

  // Combine per-thread (m, d) pairs pairwise in shared memory.
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      const float m1 = max_reduce[tid];
      const float d1 = sum_reduce[tid];
      const float m2 = max_reduce[tid + s];
      const float d2 = sum_reduce[tid + s];
      const float m_new = fmaxf(m1, m2);
      max_reduce[tid] = m_new;
      sum_reduce[tid] = d1 * expf(m1 - m_new) + d2 * expf(m2 - m_new);
    }
    __syncthreads();
  }
  const float row_max = max_reduce[0];
  const float row_sum = sum_reduce[0];
  __syncthreads();

  for (int c = tid; c < cols; c += blockDim.x) y[c] = expf(x[c] - row_max) / row_sum;
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

  float* dev_in = nullptr;
  float* dev_out = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_in), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_out), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_in, host.data(), bytes));

  constexpr double kPeakGbPerSec = 1555.0;
  const size_t bytes_moved = 2 * bytes;

  auto check = [&](const char* name, const std::function<void()>& launch) -> bool {
    launch();
    GPU_CHECK(gpuDeviceSynchronize());
    GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_out, bytes));
    if (!gklab::verifyClose(result, reference, 1.0e-3f)) {
      std::fprintf(stderr, "%s FAILED\n", name);
      return false;
    }
    gklab::report(name, gklab::benchmarkKernel(launch, bytes_moved, 0.0), kPeakGbPerSec, 0.0);
    return true;
  };

  if (!check("softmax_three_pass", [&]() { GPU_LAUNCH(softmaxThreePass, kRows, kBlockSize, 0, dev_in, dev_out, kRows, kCols); }))
    return EXIT_FAILURE;
  if (!check("softmax_online", [&]() { GPU_LAUNCH(softmaxOnline, kRows, kBlockSize, 0, dev_in, dev_out, kRows, kCols); }))
    return EXIT_FAILURE;

  GPU_CHECK(gpuFree(dev_in));
  GPU_CHECK(gpuFree(dev_out));
  return EXIT_SUCCESS;
}
