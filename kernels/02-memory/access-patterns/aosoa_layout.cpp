// aosoa_layout.cpp
//
// Benchmarks the same "scale field x of every particle" operation over three
// data layouts: AoS (stride-3 access), SoA (fully coalesced), and AoSoA
// (SoA chunks of warp size, object grouping preserved between chunks).
// Expected result: SoA and AoSoA reach similar bandwidth, AoS loses ~3x.
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kChunk = 32;  // AoSoA chunk = warp size
constexpr int kNumParticles = 1 << 24;

struct Particle {
  float x, y, z;
};

// AoSoA: each chunk stores 32 x's, then 32 y's, then 32 z's. Lane i of a warp
// reads x[i] within one chunk -> contiguous 128 bytes, coalesced like SoA.
struct ParticleChunk {
  float x[kChunk];
  float y[kChunk];
  float z[kChunk];
};

// AoS: field x lives at bytes 0, 12, 24, ... A warp reading .x touches 3x the
// segments it needs.
__global__ void scaleXAos(Particle* p, float k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i].x *= k;
}

// SoA: field x is one contiguous array, the ideal warp access.
__global__ void scaleXSoa(float* x, float k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= k;
}

// AoSoA: locate the chunk, then the lane within the chunk. The intra-chunk
// x[] array is contiguous, so a warp maps to one 128-byte segment.
__global__ void scaleXAosoa(ParticleChunk* chunks, float k, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) chunks[i / kChunk].x[i % kChunk] *= k;
}

}  // namespace

int main() {
  const int n = kNumParticles;
  const size_t x_bytes = static_cast<size_t>(n) * sizeof(float);

  std::vector<Particle> aos(n);
  std::vector<float> soa_x(n);
  std::vector<ParticleChunk> aosoa(n / kChunk);
  for (int i = 0; i < n; ++i) {
    const float v = static_cast<float>(i % 1013);
    aos[i] = {v, 2.0f * v, 3.0f * v};
    soa_x[i] = v;
    aosoa[i / kChunk].x[i % kChunk] = v;
    aosoa[i / kChunk].y[i % kChunk] = 2.0f * v;
    aosoa[i / kChunk].z[i % kChunk] = 3.0f * v;
  }

  Particle* dev_aos = nullptr;
  float* dev_soa_x = nullptr;
  ParticleChunk* dev_aosoa = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_aos), n * sizeof(Particle)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_soa_x), x_bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_aosoa), aosoa.size() * sizeof(ParticleChunk)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_aos, aos.data(), n * sizeof(Particle)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_soa_x, soa_x.data(), x_bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_aosoa, aosoa.data(), aosoa.size() * sizeof(ParticleChunk)));

  constexpr double kPeakGbPerSec = 1555.0;
  constexpr float kScale = 1.0001f;
  const int grid = (n + kBlockSize - 1) / kBlockSize;

  // Useful traffic is identical for all three layouts: read + write of x only.
  const size_t useful_bytes = 2 * x_bytes;

  auto launch_aos = [&]() { GPU_LAUNCH(scaleXAos, grid, kBlockSize, 0, dev_aos, kScale, n); };
  auto launch_soa = [&]() { GPU_LAUNCH(scaleXSoa, grid, kBlockSize, 0, dev_soa_x, kScale, n); };
  auto launch_aosoa = [&]() { GPU_LAUNCH(scaleXAosoa, grid, kBlockSize, 0, dev_aosoa, kScale, n); };

  std::printf("aosoa_layout: scale x of %d particles (useful bytes identical per layout)\n", n);
  gklab::report("aos_stride3", gklab::benchmarkKernel(launch_aos, useful_bytes, 0.0), kPeakGbPerSec, 0.0);
  gklab::report("soa", gklab::benchmarkKernel(launch_soa, useful_bytes, 0.0), kPeakGbPerSec, 0.0);
  gklab::report("aosoa_chunk32", gklab::benchmarkKernel(launch_aosoa, useful_bytes, 0.0), kPeakGbPerSec, 0.0);

  GPU_CHECK(gpuFree(dev_aos));
  GPU_CHECK(gpuFree(dev_soa_x));
  GPU_CHECK(gpuFree(dev_aosoa));
  return EXIT_SUCCESS;
}
