// stream_pipeline.cpp
//
// Chapter 28 reference solution: compare a single-stream host-device pipeline
// with a two-stream double-buffered pipeline using pinned host memory and
// asynchronous copies.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;
constexpr float kScale = 3.0f;

__global__ void scaleKernel(const float* in, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * kScale;
}

double secondsSince(std::chrono::steady_clock::time_point start) {
  auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(stop - start).count();
}

bool verify(const char* name, const std::vector<float>& got, const std::vector<float>& input) {
  double maxRel = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    float want = input[i] * kScale;
    double rel = std::fabs(got[i] - want) / std::max(1.0e-6f, std::fabs(want));
    maxRel = std::max(maxRel, rel);
  }
  if (maxRel > 1.0e-6) {
    std::fprintf(stderr, "%s FAILED: max relative error %.3e\n", name, maxRel);
    return false;
  }
  std::printf("%s correctness OK (max relative error %.3e)\n", name, maxRel);
  return true;
}

}  // namespace

int main() {
  constexpr int kN = 1 << 24;
  constexpr int kChunkElems = 1 << 20;
  static_assert(kN % kChunkElems == 0, "example uses fixed-size chunks");
  constexpr int kChunks = kN / kChunkElems;
  const size_t chunkBytes = static_cast<size_t>(kChunkElems) * sizeof(float);
  const size_t totalBytes = static_cast<size_t>(kN) * sizeof(float);

  std::vector<float> input(kN);
  std::vector<float> result(kN);
  for (int i = 0; i < kN; ++i) input[i] = static_cast<float>(i % 251) * 0.25f;

  float* devIn = nullptr;
  float* devOut = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devIn), chunkBytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devOut), chunkBytes));

  auto t0 = std::chrono::steady_clock::now();
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const float* src = input.data() + static_cast<size_t>(chunk) * kChunkElems;
    float* dst = result.data() + static_cast<size_t>(chunk) * kChunkElems;
    GPU_CHECK(gpuMemcpyHostToDevice(devIn, src, chunkBytes));
    int grid = (kChunkElems + kBlockSize - 1) / kBlockSize;
    GPU_LAUNCH(scaleKernel, grid, kBlockSize, 0, devIn, devOut, kChunkElems);
    GPU_CHECK(gpuMemcpyDeviceToHost(dst, devOut, chunkBytes));
  }
  GPU_CHECK(gpuDeviceSynchronize());
  double syncSeconds = secondsSince(t0);
  if (!verify("stream_sync", result, input)) return EXIT_FAILURE;

  GPU_CHECK(gpuFree(devIn));
  GPU_CHECK(gpuFree(devOut));

  float* hostIn[2] = {nullptr, nullptr};
  float* hostOut[2] = {nullptr, nullptr};
  float* pipeIn[2] = {nullptr, nullptr};
  float* pipeOut[2] = {nullptr, nullptr};
  GpuStream streams[2]{};
  for (int s = 0; s < 2; ++s) {
    GPU_CHECK(gpuMallocHost(reinterpret_cast<void**>(&hostIn[s]), chunkBytes));
    GPU_CHECK(gpuMallocHost(reinterpret_cast<void**>(&hostOut[s]), chunkBytes));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&pipeIn[s]), chunkBytes));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&pipeOut[s]), chunkBytes));
    GPU_CHECK(gpuStreamCreate(&streams[s]));
  }

  std::fill(result.begin(), result.end(), 0.0f);
  t0 = std::chrono::steady_clock::now();
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    int slot = chunk & 1;
    if (chunk >= 2) {
      GPU_CHECK(gpuStreamSynchronize(streams[slot]));
      std::copy(hostOut[slot], hostOut[slot] + kChunkElems,
                result.data() + static_cast<size_t>(chunk - 2) * kChunkElems);
    }

    std::copy(input.data() + static_cast<size_t>(chunk) * kChunkElems,
              input.data() + static_cast<size_t>(chunk + 1) * kChunkElems, hostIn[slot]);
    GPU_CHECK(gpuMemcpyHostToDeviceAsync(pipeIn[slot], hostIn[slot], chunkBytes, streams[slot]));
    int grid = (kChunkElems + kBlockSize - 1) / kBlockSize;
    GPU_LAUNCH_STREAM(scaleKernel, grid, kBlockSize, 0, streams[slot], pipeIn[slot], pipeOut[slot],
                      kChunkElems);
    GPU_CHECK(gpuMemcpyDeviceToHostAsync(hostOut[slot], pipeOut[slot], chunkBytes, streams[slot]));
  }
  for (int s = 0; s < 2; ++s) {
    GPU_CHECK(gpuStreamSynchronize(streams[s]));
    int chunk = kChunks - 2 + s;
    std::copy(hostOut[s], hostOut[s] + kChunkElems,
              result.data() + static_cast<size_t>(chunk) * kChunkElems);
  }
  double pipeSeconds = secondsSince(t0);
  if (!verify("stream_double_buffered", result, input)) return EXIT_FAILURE;

  const double gb = static_cast<double>(2 * totalBytes) / 1.0e9;
  std::printf("%-28s %8.3f ms | %8.1f GB/s end-to-end\n", "stream_sync", syncSeconds * 1.0e3,
              gb / syncSeconds);
  std::printf("%-28s %8.3f ms | %8.1f GB/s end-to-end\n", "stream_double_buffered",
              pipeSeconds * 1.0e3, gb / pipeSeconds);

  for (int s = 0; s < 2; ++s) {
    GPU_CHECK(gpuStreamDestroy(streams[s]));
    GPU_CHECK(gpuFree(pipeIn[s]));
    GPU_CHECK(gpuFree(pipeOut[s]));
    GPU_CHECK(gpuFreeHost(hostIn[s]));
    GPU_CHECK(gpuFreeHost(hostOut[s]));
  }
  return EXIT_SUCCESS;
}
