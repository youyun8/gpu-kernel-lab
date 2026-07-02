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
__global__ void copyPipelined(const float* in, float* out, int n, int tilesPerBlock) {
  __shared__ float buf[2][kTile];
  int tid = threadIdx.x;
  int blockStart = blockIdx.x * tilesPerBlock * kTile;

  auto load = [&](int tileIdx, int stage) {
    int g = blockStart + tileIdx * kTile + tid;
    buf[stage][tid] = (g < n) ? in[g] : 0.0f;
  };

  int stage = 0;
  load(0, stage);
  __syncthreads();

  for (int t = 0; t < tilesPerBlock; ++t) {
    int nextStage = stage ^ 1;
    if (t + 1 < tilesPerBlock) load(t + 1, nextStage);  // prefetch
    int g = blockStart + t * kTile + tid;
    if (g < n) out[g] = buf[stage][tid] * kScale;
    __syncthreads();
    stage = nextStage;
  }
}

}  // namespace

int main() {
  constexpr int kN = 1 << 24;
  const size_t bytes = static_cast<size_t>(kN) * sizeof(float);
  constexpr int kTilesPerBlock = 8;

  std::vector<float> host(kN);
  std::vector<float> reference(kN);
  for (int i = 0; i < kN; ++i) {
    host[i] = static_cast<float>(i % 251) * 0.3f;
    reference[i] = host[i] * kScale;
  }
  std::vector<float> result(kN);

  float* devIn = nullptr;
  float* devOut = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devIn), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devOut), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(devIn, host.data(), bytes));

  constexpr double kPeakGbPerSec = 1555.0;
  const size_t bytesMoved = 2 * bytes;

  const int gridSync = (kN + kBlockSize - 1) / kBlockSize;
  auto launchSync = [&]() { GPU_LAUNCH(copySync, gridSync, kBlockSize, 0, devIn, devOut, kN); };
  launchSync();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devOut, bytes));
  if (!gklab::verifyClose(result, reference)) return EXIT_FAILURE;
  gklab::report("copy_sync", gklab::benchmarkKernel(launchSync, bytesMoved, 0.0), kPeakGbPerSec, 0.0);

  const int elementsPerBlock = kTilesPerBlock * kTile;
  const int gridPipe = (kN + elementsPerBlock - 1) / elementsPerBlock;
  auto launchPipe = [&]() { GPU_LAUNCH(copyPipelined, gridPipe, kBlockSize, 0, devIn, devOut, kN, kTilesPerBlock); };
  launchPipe();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devOut, bytes));
  if (!gklab::verifyClose(result, reference)) return EXIT_FAILURE;
  gklab::report("copy_pipelined", gklab::benchmarkKernel(launchPipe, bytesMoved, 0.0), kPeakGbPerSec, 0.0);

  GPU_CHECK(gpuFree(devIn));
  GPU_CHECK(gpuFree(devOut));
  return EXIT_SUCCESS;
}
