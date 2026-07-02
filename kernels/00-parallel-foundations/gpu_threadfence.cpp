// gpu_threadfence.cpp
//
// Single-pass reduction using the classic "last block finishes the job"
// pattern, which needs two things blocks normally do not have between them:
//   1. __threadfence() - make this block's partial sum visible to the whole
//      device *before* announcing completion, and
//   2. an atomic ticket counter - the block that takes the last ticket knows
//      every other partial is already published, so it can safely reduce them.
// Without the fence, the last block could read a stale partial: a cross-block
// data race that no __syncthreads() can fix (barriers only cover one block).
#include <cstdio>
#include <vector>

#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kNumItems = 1 << 24;

__device__ unsigned int retired_blocks = 0;

__global__ void singlePassReduce(const float* in, float* partials, float* out, int n) {
  __shared__ float shared_partials[kBlockSize];
  __shared__ bool is_last_block;

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  shared_partials[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();

  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) shared_partials[tid] += shared_partials[tid + stride];
    __syncthreads();
  }

  if (tid == 0) {
    partials[blockIdx.x] = shared_partials[0];
    // Publish the partial to every block before taking a ticket. Without this
    // fence the atomic below could become visible first, and the last block
    // would read a stale (or uninitialized) partials[blockIdx.x].
    __threadfence();
    unsigned int ticket = atomicAdd(&retired_blocks, 1u);
    is_last_block = (ticket == gridDim.x - 1);
  }
  __syncthreads();

  if (is_last_block) {
    // Grid-stride loop over all published partials, then one final tree.
    float sum = 0.0f;
    for (int b = tid; b < static_cast<int>(gridDim.x); b += kBlockSize) {
      sum += partials[b];
    }
    shared_partials[tid] = sum;
    __syncthreads();
    for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
      if (tid < stride) shared_partials[tid] += shared_partials[tid + stride];
      __syncthreads();
    }
    if (tid == 0) {
      *out = shared_partials[0];
      retired_blocks = 0;  // reset so the kernel can be launched again
    }
  }
}

}  // namespace

int main() {
  std::vector<float> host(kNumItems);
  double reference = 0.0;
  for (int i = 0; i < kNumItems; ++i) {
    host[i] = static_cast<float>((i % 7) - 3);
    reference += host[i];
  }

  const int grid = (kNumItems + kBlockSize - 1) / kBlockSize;
  float* dev_in = nullptr;
  float* dev_partials = nullptr;
  float* dev_out = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_in), kNumItems * sizeof(float)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_partials), grid * sizeof(float)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_out), sizeof(float)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_in, host.data(), kNumItems * sizeof(float)));

  GPU_LAUNCH(singlePassReduce, grid, kBlockSize, 0, dev_in, dev_partials, dev_out, kNumItems);
  GPU_CHECK(gpuDeviceSynchronize());

  float result = 0.0f;
  GPU_CHECK(gpuMemcpyDeviceToHost(&result, dev_out, sizeof(float)));
  std::printf("gpu_threadfence: single-pass reduce got %.1f, reference %.1f -> %s\n",
              result, reference, static_cast<double>(result) == reference ? "OK" : "MISMATCH");

  GPU_CHECK(gpuFree(dev_in));
  GPU_CHECK(gpuFree(dev_partials));
  GPU_CHECK(gpuFree(dev_out));
  return 0;
}
