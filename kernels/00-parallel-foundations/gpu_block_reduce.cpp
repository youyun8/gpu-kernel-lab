// gpu_block_reduce.cpp
//
// The GPU counterpart of OpenMP's reduction clause: instead of every thread
// adding into one shared counter, each block privatizes work into shared
// memory, tree-reduces it with __syncthreads() between rounds, and only the
// per-block result touches global memory (one atomicAdd per block).
// __syncthreads() is the barrier that makes the shared-memory tree safe: it
// guarantees every partial from round k is written before round k+1 reads it.
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kNumItems = 1 << 24;

__global__ void blockReduceSum(const float* in, float* out, int n) {
  __shared__ float partials[kBlockSize];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int tid = threadIdx.x;
  partials[tid] = (i < n) ? in[i] : 0.0f;
  __syncthreads();  // all loads into shared memory complete before the tree

  // Tree reduction: stride halves each round. The barrier between rounds is
  // what prevents a data race between the writer of partials[tid] and the
  // reader of partials[tid + stride] in the next round.
  for (int stride = kBlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partials[tid] += partials[tid + stride];
    }
    __syncthreads();
  }

  // One atomic per block instead of one per thread: 256x less contention.
  if (tid == 0) {
    atomicAdd(out, partials[0]);
  }
}

}  // namespace

int main() {
  std::vector<float> host(kNumItems);
  double reference = 0.0;
  for (int i = 0; i < kNumItems; ++i) {
    host[i] = static_cast<float>((i % 7) - 3);  // sums stay small and exact
    reference += host[i];
  }

  float* dev_in = nullptr;
  float* dev_out = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_in), kNumItems * sizeof(float)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_out), sizeof(float)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_in, host.data(), kNumItems * sizeof(float)));

  const int grid = (kNumItems + kBlockSize - 1) / kBlockSize;
  auto launch = [&]() {
    GPU_CHECK(gpuMemset(dev_out, 0, sizeof(float)));
    GPU_LAUNCH(blockReduceSum, grid, kBlockSize, 0, dev_in, dev_out, kNumItems);
  };

  launch();
  GPU_CHECK(gpuDeviceSynchronize());
  float result = 0.0f;
  GPU_CHECK(gpuMemcpyDeviceToHost(&result, dev_out, sizeof(float)));
  std::printf("gpu_block_reduce: got %.1f, reference %.1f -> %s\n", result, reference,
              static_cast<double>(result) == reference ? "OK" : "MISMATCH");

  gklab::report("block_reduce", gklab::benchmarkKernel(launch, kNumItems * sizeof(float), kNumItems),
                1555.0, 19500.0);

  GPU_CHECK(gpuFree(dev_in));
  GPU_CHECK(gpuFree(dev_out));
  return 0;
}
