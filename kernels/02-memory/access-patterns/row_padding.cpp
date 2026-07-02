// row_padding.cpp
//
// Shows why the leading dimension of a 2D tensor should be padded to a
// multiple of 128 bytes. With ld = 1000 floats (4000 bytes) every row starts
// at a different offset within a 128-byte segment, so warps reading rows keep
// straddling segment boundaries. Padding to ld = 1024 floats wastes 2.4% of
// memory and makes every row start aligned.
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kRows = 8192;
constexpr int kCols = 1000;          // 4000 bytes per row: NOT 128-byte aligned
constexpr int kPaddedLd = 1024;      // 4096 bytes per row: 128-byte aligned

// One block strip-mines one row. The access is stride-1 within the row, but
// whether the row STARTS on a segment boundary depends entirely on ld.
__global__ void scaleRows(float* data, int ld, int rows, int cols, float k) {
  float* row = data + static_cast<size_t>(blockIdx.x) * ld;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) row[c] *= k;
}

}  // namespace

int main() {
  const size_t tight_elems = static_cast<size_t>(kRows) * kCols;
  const size_t padded_elems = static_cast<size_t>(kRows) * kPaddedLd;

  std::vector<float> host(padded_elems, 1.0f);

  float* dev_tight = nullptr;
  float* dev_padded = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_tight), tight_elems * sizeof(float)));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_padded), padded_elems * sizeof(float)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_tight, host.data(), tight_elems * sizeof(float)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_padded, host.data(), padded_elems * sizeof(float)));

  constexpr double kPeakGbPerSec = 1555.0;
  constexpr float kScale = 1.0001f;
  // Useful traffic is identical: read + write of rows*cols elements. The
  // padded layout moves the same useful bytes, just from aligned addresses.
  const size_t moved = 2 * tight_elems * sizeof(float);

  auto launch_tight = [&]() { GPU_LAUNCH(scaleRows, kRows, kBlockSize, 0, dev_tight, kCols, kRows, kCols, kScale); };
  auto launch_padded = [&]() { GPU_LAUNCH(scaleRows, kRows, kBlockSize, 0, dev_padded, kPaddedLd, kRows, kCols, kScale); };

  std::printf("row_padding: %d rows x %d cols, ld=%d (unaligned rows) vs ld=%d (128B-aligned)\n",
              kRows, kCols, kCols, kPaddedLd);
  gklab::report("ld_1000_unaligned", gklab::benchmarkKernel(launch_tight, moved, 0.0), kPeakGbPerSec, 0.0);
  gklab::report("ld_1024_padded", gklab::benchmarkKernel(launch_padded, moved, 0.0), kPeakGbPerSec, 0.0);

  GPU_CHECK(gpuFree(dev_tight));
  GPU_CHECK(gpuFree(dev_padded));
  return EXIT_SUCCESS;
}
