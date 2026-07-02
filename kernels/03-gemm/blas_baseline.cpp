// blas_baseline.cpp
//
// Calls the vendor BLAS (cuBLAS / hipBLAS) for the same SGEMM the hand-written
// sgemm_step* kernels compute, so you can see how far the tutorial kernels are
// from a production library (chapter 15). Only built when the BLAS library is
// found; see this directory's CMakeLists.txt.
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

#if defined(USE_HIP)
#include <hipblas/hipblas.h>
using BlasHandle = hipblasHandle_t;
#define BLAS_CHECK(expr)                                                   \
  do {                                                                     \
    if ((expr) != HIPBLAS_STATUS_SUCCESS) {                                \
      std::fprintf(stderr, "hipBLAS error at %s:%d\n", __FILE__, __LINE__); \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)
#else
#include <cublas_v2.h>
using BlasHandle = cublasHandle_t;
#define BLAS_CHECK(expr)                                                   \
  do {                                                                     \
    if ((expr) != CUBLAS_STATUS_SUCCESS) {                                 \
      std::fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__);  \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)
#endif

namespace {

constexpr int kDim = 2048;  // C = A * B, all kDim x kDim, column-major for BLAS

}  // namespace

int main() {
  const size_t elems = static_cast<size_t>(kDim) * kDim;
  const size_t bytes = elems * sizeof(float);

  std::vector<float> host_a(elems), host_b(elems);
  for (size_t i = 0; i < elems; ++i) {
    host_a[i] = static_cast<float>(i % 13) * 0.1f;
    host_b[i] = static_cast<float>(i % 7) * 0.1f;
  }

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_a), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_b), bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_c), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_a, host_a.data(), bytes));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_b, host_b.data(), bytes));

  BlasHandle handle;
  const float alpha = 1.0f;
  const float beta = 0.0f;

#if defined(USE_HIP)
  BLAS_CHECK(hipblasCreate(&handle));
  auto launch = [&]() {
    BLAS_CHECK(hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, kDim, kDim, kDim,
                            &alpha, dev_a, kDim, dev_b, kDim, &beta, dev_c, kDim));
  };
#else
  BLAS_CHECK(cublasCreate(&handle));
  auto launch = [&]() {
    BLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, kDim, kDim, kDim,
                           &alpha, dev_a, kDim, dev_b, kDim, &beta, dev_c, kDim));
  };
#endif

  launch();
  GPU_CHECK(gpuDeviceSynchronize());

  constexpr double kPeakGbPerSec = 1555.0;
  constexpr double kPeakGflopPerSec = 19500.0;
  const double flops = 2.0 * static_cast<double>(kDim) * kDim * kDim;
  const size_t moved = 3 * bytes;  // A + B read, C written (ignoring reuse)
  gklab::report("blas_sgemm_2048", gklab::benchmarkKernel(launch, moved, flops),
                kPeakGbPerSec, kPeakGflopPerSec);
  std::printf("compare against sgemm_step0..step4 to see the remaining gap\n");

#if defined(USE_HIP)
  BLAS_CHECK(hipblasDestroy(handle));
#else
  BLAS_CHECK(cublasDestroy(handle));
#endif
  GPU_CHECK(gpuFree(dev_a));
  GPU_CHECK(gpuFree(dev_b));
  GPU_CHECK(gpuFree(dev_c));
  return EXIT_SUCCESS;
}
