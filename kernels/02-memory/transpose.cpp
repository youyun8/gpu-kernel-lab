// transpose.cpp
//
// Matrix transpose in three stages (chapter 7): naive, shared-memory tiled, and
// padded (bank-conflict-free). Each is validated against a CPU transpose and
// benchmarked for effective bandwidth.
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kTile = 32;

// Stage 1: naive. Coalesced read, strided (uncoalesced) write.
__global__ void transposeNaive(const float* in, float* out, int n) {
  int x = blockIdx.x * kTile + threadIdx.x;
  int y = blockIdx.y * kTile + threadIdx.y;
  if (x < n && y < n) {
    out[x * n + y] = in[y * n + x];
  }
}

// Stage 2: shared-memory tile. Both global accesses coalesced, but the [32][32]
// tile causes bank conflicts on the transposed shared-memory read.
__global__ void transposeShared(const float* in, float* out, int n) {
  __shared__ float tile[kTile][kTile];
  int x = blockIdx.x * kTile + threadIdx.x;
  int y = blockIdx.y * kTile + threadIdx.y;
  if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y * n + x];
  __syncthreads();
  int tx = blockIdx.y * kTile + threadIdx.x;
  int ty = blockIdx.x * kTile + threadIdx.y;
  if (tx < n && ty < n) out[ty * n + tx] = tile[threadIdx.x][threadIdx.y];
}

// Stage 3: padded tile removes the bank conflict via a +1 column.
__global__ void transposePadded(const float* in, float* out, int n) {
  __shared__ float tile[kTile][kTile + 1];
  int x = blockIdx.x * kTile + threadIdx.x;
  int y = blockIdx.y * kTile + threadIdx.y;
  if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y * n + x];
  __syncthreads();
  int tx = blockIdx.y * kTile + threadIdx.x;
  int ty = blockIdx.x * kTile + threadIdx.y;
  if (tx < n && ty < n) out[ty * n + tx] = tile[threadIdx.x][threadIdx.y];
}

void cpuTranspose(const std::vector<float>& in, std::vector<float>& out, int n) {
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < n; ++c) out[c * n + r] = in[r * n + c];
}

}  // namespace

int main() {
  constexpr int kN = 2048;  // divisible by kTile
  const size_t bytes = static_cast<size_t>(kN) * kN * sizeof(float);

  std::vector<float> host(static_cast<size_t>(kN) * kN);
  std::vector<float> reference(static_cast<size_t>(kN) * kN);
  std::vector<float> result(static_cast<size_t>(kN) * kN);
  for (size_t i = 0; i < host.size(); ++i) host[i] = static_cast<float>(i % 251) * 0.5f;
  cpuTranspose(host, reference, kN);

  float* devIn = nullptr;
  float* devOut = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devIn), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devOut), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(devIn, host.data(), bytes));

  dim3 block(kTile, kTile);
  dim3 grid(kN / kTile, kN / kTile);
  constexpr double kPeakGbPerSec = 1555.0;

  struct Variant {
    const char* name;
    void (*kernel)(const float*, float*, int);
  };
  const Variant variants[] = {
      {"transpose_naive", transposeNaive},
      {"transpose_shared", transposeShared},
      {"transpose_padded", transposePadded},
  };

  for (const auto& v : variants) {
    auto launch = [&]() { GPU_LAUNCH(v.kernel, grid, block, 0, devIn, devOut, kN); };
    launch();
    GPU_CHECK(gpuDeviceSynchronize());
    GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devOut, bytes));
    if (!gklab::verifyClose(result, reference)) return EXIT_FAILURE;
    gklab::report(v.name, gklab::benchmarkKernel(launch, 2 * bytes, 0.0), kPeakGbPerSec, 0.0);
  }

  GPU_CHECK(gpuFree(devIn));
  GPU_CHECK(gpuFree(devOut));
  return EXIT_SUCCESS;
}
