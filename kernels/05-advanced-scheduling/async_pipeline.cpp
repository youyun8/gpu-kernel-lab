// async_pipeline.cpp
//
// A simplified two-stage producer/consumer copy pipeline (chapter 14). It stages
// tiles through shared memory with double buffering so the load of the next tile
// overlaps with the "compute" (here a scale) on the current tile. Compared with
// a straightforward synchronous copy. The portable version uses plain shared
// memory; on CUDA Ampere+ the loads would use cp.async (see the chapter).
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kTile = 256;  // elements staged per iteration (== block size)
constexpr float kScale = 2.0f;

// Straightforward: read, scale, write, one element per thread.
__global__ void copySync(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * kScale;
}

// Pipelined: each block streams its chunk through two shared-memory buffers,
// prefetching the next tile while scaling the current one.
__global__ void copyPipelined(const float* in, float* out, int n, int tiles_per_block) {
  __shared__ float buf[2][kTile];
  int tid = threadIdx.x;
  int block_start = blockIdx.x * tiles_per_block * kTile;

  auto load = [&](int tile_idx, int stage) {
    int g = block_start + tile_idx * kTile + tid;
    buf[stage][tid] = (g < n) ? in[g] : 0.0f;
  };

  int stage = 0;
  load(0, stage);
  __syncthreads();

  for (int t = 0; t < tiles_per_block; ++t) {
    int next_stage = stage ^ 1;
    if (t + 1 < tiles_per_block) load(t + 1, next_stage);  // prefetch
    int g = block_start + t * kTile + tid;
    if (g < n) out[g] = buf[stage][tid] * kScale;
    __syncthreads();
    stage = next_stage;
  }
}

}  // namespace

int main() {
  constexpr int kSizeN = 1 << 24;
  const size_t bytes = static_cast<size_t>(kSizeN) * sizeof(float);
  constexpr int kTilesPerBlock = 8;

  std::vector<float> host(kSizeN);
  std::vector<float> reference(kSizeN);
  for (int i = 0; i < kSizeN; ++i) {
    host[i] = static_cast<float>(i % 251) * 0.3f;
    reference[i] = host[i] * kScale;
  }
  std::vector<float> result(kSizeN);

  float* dev_in = nullptr;
  float* dev_out = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_in), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_out), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_in, host.data(), bytes));

  constexpr double kPeakGbPerSec = 1555.0;
  const size_t bytes_moved = 2 * bytes;

  const int grid_sync = (kSizeN + kBlockSize - 1) / kBlockSize;
  auto launch_sync = [&]() { GPU_LAUNCH(copySync, grid_sync, kBlockSize, 0, dev_in, dev_out, kSizeN); };
  launch_sync();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_out, bytes));
  if (!gklab::verifyClose(result, reference)) return EXIT_FAILURE;
  gklab::report("copy_sync", gklab::benchmarkKernel(launch_sync, bytes_moved, 0.0), kPeakGbPerSec, 0.0);

  const int elements_per_block = kTilesPerBlock * kTile;
  const int grid_pipe = (kSizeN + elements_per_block - 1) / elements_per_block;
  auto launch_pipe = [&]() { GPU_LAUNCH(copyPipelined, grid_pipe, kBlockSize, 0, dev_in, dev_out, kSizeN, kTilesPerBlock); };
  launch_pipe();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_out, bytes));
  if (!gklab::verifyClose(result, reference)) return EXIT_FAILURE;
  gklab::report("copy_pipelined", gklab::benchmarkKernel(launch_pipe, bytes_moved, 0.0), kPeakGbPerSec, 0.0);

  GPU_CHECK(gpuFree(dev_in));
  GPU_CHECK(gpuFree(dev_out));
  return EXIT_SUCCESS;
}
