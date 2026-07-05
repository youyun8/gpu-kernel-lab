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
  double max_rel = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    float want = input[i] * kScale;
    double rel = std::fabs(got[i] - want) / std::max(1.0e-6f, std::fabs(want));
    max_rel = std::max(max_rel, rel);
  }
  if (max_rel > 1.0e-6) {
    std::fprintf(stderr, "%s FAILED: max relative error %.3e\n", name, max_rel);
    return false;
  }
  std::printf("%s correctness OK (max relative error %.3e)\n", name, max_rel);
  return true;
}

}  // namespace

int main() {
  constexpr int kSizeN = 1 << 24;
  constexpr int kChunkElems = 1 << 20;
  static_assert(kSizeN % kChunkElems == 0, "example uses fixed-size chunks");
  constexpr int kChunks = kSizeN / kChunkElems;
  const size_t chunk_bytes = static_cast<size_t>(kChunkElems) * sizeof(float);
  const size_t total_bytes = static_cast<size_t>(kSizeN) * sizeof(float);

  std::vector<float> input(kSizeN);
  std::vector<float> result(kSizeN);
  for (int i = 0; i < kSizeN; ++i) input[i] = static_cast<float>(i % 251) * 0.25f;

  float* dev_in = nullptr;
  float* dev_out = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_in), chunk_bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_out), chunk_bytes));

  auto t0 = std::chrono::steady_clock::now();
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    const float* src = input.data() + static_cast<size_t>(chunk) * kChunkElems;
    float* dst = result.data() + static_cast<size_t>(chunk) * kChunkElems;
    GPU_CHECK(gpuMemcpyHostToDevice(dev_in, src, chunk_bytes));
    int grid = (kChunkElems + kBlockSize - 1) / kBlockSize;
    GPU_LAUNCH(scaleKernel, grid, kBlockSize, 0, dev_in, dev_out, kChunkElems);
    GPU_CHECK(gpuMemcpyDeviceToHost(dst, dev_out, chunk_bytes));
  }
  GPU_CHECK(gpuDeviceSynchronize());
  double sync_seconds = secondsSince(t0);
  if (!verify("stream_sync", result, input)) return EXIT_FAILURE;

  GPU_CHECK(gpuFree(dev_in));
  GPU_CHECK(gpuFree(dev_out));

  float* host_in[2] = {nullptr, nullptr};
  float* host_out[2] = {nullptr, nullptr};
  float* pipe_in[2] = {nullptr, nullptr};
  float* pipe_out[2] = {nullptr, nullptr};
  GpuStream streams[2]{};
  for (int s = 0; s < 2; ++s) {
    GPU_CHECK(gpuMallocHost(reinterpret_cast<void**>(&host_in[s]), chunk_bytes));
    GPU_CHECK(gpuMallocHost(reinterpret_cast<void**>(&host_out[s]), chunk_bytes));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&pipe_in[s]), chunk_bytes));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&pipe_out[s]), chunk_bytes));
    GPU_CHECK(gpuStreamCreate(&streams[s]));
  }

  std::fill(result.begin(), result.end(), 0.0f);
  t0 = std::chrono::steady_clock::now();
  for (int chunk = 0; chunk < kChunks; ++chunk) {
    int slot = chunk & 1;
    if (chunk >= 2) {
      GPU_CHECK(gpuStreamSynchronize(streams[slot]));
      std::copy(host_out[slot], host_out[slot] + kChunkElems,
                result.data() + static_cast<size_t>(chunk - 2) * kChunkElems);
    }

    std::copy(input.data() + static_cast<size_t>(chunk) * kChunkElems,
              input.data() + static_cast<size_t>(chunk + 1) * kChunkElems, host_in[slot]);
    GPU_CHECK(gpuMemcpyHostToDeviceAsync(pipe_in[slot], host_in[slot], chunk_bytes, streams[slot]));
    int grid = (kChunkElems + kBlockSize - 1) / kBlockSize;
    GPU_LAUNCH_STREAM(scaleKernel, grid, kBlockSize, 0, streams[slot], pipe_in[slot], pipe_out[slot],
                      kChunkElems);
    GPU_CHECK(gpuMemcpyDeviceToHostAsync(host_out[slot], pipe_out[slot], chunk_bytes, streams[slot]));
  }
  for (int s = 0; s < 2; ++s) {
    GPU_CHECK(gpuStreamSynchronize(streams[s]));
    int chunk = kChunks - 2 + s;
    std::copy(host_out[s], host_out[s] + kChunkElems,
              result.data() + static_cast<size_t>(chunk) * kChunkElems);
  }
  double pipe_seconds = secondsSince(t0);
  if (!verify("stream_double_buffered", result, input)) return EXIT_FAILURE;

  const double gb = static_cast<double>(2 * total_bytes) / 1.0e9;
  std::printf("%-28s %8.3f ms | %8.1f GB/s end-to-end\n", "stream_sync", sync_seconds * 1.0e3,
              gb / sync_seconds);
  std::printf("%-28s %8.3f ms | %8.1f GB/s end-to-end\n", "stream_double_buffered",
              pipe_seconds * 1.0e3, gb / pipe_seconds);

  for (int s = 0; s < 2; ++s) {
    GPU_CHECK(gpuStreamDestroy(streams[s]));
    GPU_CHECK(gpuFree(pipe_in[s]));
    GPU_CHECK(gpuFree(pipe_out[s]));
    GPU_CHECK(gpuFreeHost(host_in[s]));
    GPU_CHECK(gpuFreeHost(host_out[s]));
  }
  return EXIT_SUCCESS;
}
