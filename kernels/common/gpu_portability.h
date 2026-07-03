// gpu_portability.h
//
// A thin portability layer so the same kernel source compiles under either the
// CUDA (nvcc) or HIP (hipcc) toolchains. The build system defines one of:
//   USE_CUDA  -> include CUDA runtime, map GpuXxx -> cudaXxx
//   USE_HIP   -> include HIP runtime,  map GpuXxx -> hipXxx
// When neither GPU toolchain is available the file can still be parsed for
// syntax-only checks (see kernels/common/README.md).
#pragma once

#include <cstdio>
#include <cstdlib>

#if defined(USE_HIP)
#include <hip/hip_runtime.h>

using GpuError = hipError_t;
using GpuEvent = hipEvent_t;
using GpuStream = hipStream_t;

constexpr GpuError kGpuSuccess = hipSuccess;

#define GPU_HOST_DEVICE __host__ __device__

inline GpuError gpuMalloc(void** ptr, size_t bytes) { return hipMalloc(ptr, bytes); }
inline GpuError gpuFree(void* ptr) { return hipFree(ptr); }
inline GpuError gpuMemcpyHostToDevice(void* dst, const void* src, size_t bytes) {
  return hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
}
inline GpuError gpuMemcpyDeviceToHost(void* dst, const void* src, size_t bytes) {
  return hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
}
inline GpuError gpuMemcpyHostToDeviceAsync(void* dst, const void* src, size_t bytes, GpuStream stream) {
  return hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream);
}
inline GpuError gpuMemcpyDeviceToHostAsync(void* dst, const void* src, size_t bytes, GpuStream stream) {
  return hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream);
}
inline GpuError gpuMemset(void* ptr, int value, size_t bytes) { return hipMemset(ptr, value, bytes); }
inline GpuError gpuDeviceSynchronize() { return hipDeviceSynchronize(); }
inline GpuError gpuGetLastError() { return hipGetLastError(); }
inline const char* gpuGetErrorString(GpuError err) { return hipGetErrorString(err); }
inline GpuError gpuMallocHost(void** ptr, size_t bytes) { return hipHostMalloc(ptr, bytes); }
inline GpuError gpuFreeHost(void* ptr) { return hipHostFree(ptr); }
inline GpuError gpuStreamCreate(GpuStream* stream) { return hipStreamCreate(stream); }
inline GpuError gpuStreamDestroy(GpuStream stream) { return hipStreamDestroy(stream); }
inline GpuError gpuStreamSynchronize(GpuStream stream) { return hipStreamSynchronize(stream); }
inline GpuError gpuGetMultiprocessorCount(int* count) {
  return hipDeviceGetAttribute(count, hipDeviceAttributeMultiprocessorCount, 0);
}
inline GpuError gpuGetMaxThreadsPerMultiprocessor(int* count) {
  return hipDeviceGetAttribute(count, hipDeviceAttributeMaxThreadsPerMultiProcessor, 0);
}
template <typename KernelFn>
inline GpuError gpuOccupancyMaxActiveBlocksPerMultiprocessor(int* num_blocks, KernelFn kernel,
                                                             int block_size, size_t dynamic_smem) {
  return hipOccupancyMaxActiveBlocksPerMultiprocessor(num_blocks, kernel, block_size, dynamic_smem);
}
inline GpuError gpuSetDevice(int device) { return hipSetDevice(device); }
inline GpuError gpuGetDeviceCount(int* count) { return hipGetDeviceCount(count); }

inline GpuError gpuEventCreate(GpuEvent* event) { return hipEventCreate(event); }
inline GpuError gpuEventDestroy(GpuEvent event) { return hipEventDestroy(event); }
inline GpuError gpuEventRecord(GpuEvent event) { return hipEventRecord(event, 0); }
inline GpuError gpuEventRecordOnStream(GpuEvent event, GpuStream stream) {
  return hipEventRecord(event, stream);
}
inline GpuError gpuEventSynchronize(GpuEvent event) { return hipEventSynchronize(event); }
inline GpuError gpuEventElapsedTime(float* ms, GpuEvent start, GpuEvent stop) {
  return hipEventElapsedTime(ms, start, stop);
}

#elif defined(USE_CUDA)
#include <cuda_runtime.h>

using GpuError = cudaError_t;
using GpuEvent = cudaEvent_t;
using GpuStream = cudaStream_t;

constexpr GpuError kGpuSuccess = cudaSuccess;

#define GPU_HOST_DEVICE __host__ __device__

inline GpuError gpuMalloc(void** ptr, size_t bytes) { return cudaMalloc(ptr, bytes); }
inline GpuError gpuFree(void* ptr) { return cudaFree(ptr); }
inline GpuError gpuMemcpyHostToDevice(void* dst, const void* src, size_t bytes) {
  return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}
inline GpuError gpuMemcpyDeviceToHost(void* dst, const void* src, size_t bytes) {
  return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}
inline GpuError gpuMemcpyHostToDeviceAsync(void* dst, const void* src, size_t bytes, GpuStream stream) {
  return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
}
inline GpuError gpuMemcpyDeviceToHostAsync(void* dst, const void* src, size_t bytes, GpuStream stream) {
  return cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
}
inline GpuError gpuMemset(void* ptr, int value, size_t bytes) { return cudaMemset(ptr, value, bytes); }
inline GpuError gpuDeviceSynchronize() { return cudaDeviceSynchronize(); }
inline GpuError gpuGetLastError() { return cudaGetLastError(); }
inline const char* gpuGetErrorString(GpuError err) { return cudaGetErrorString(err); }
inline GpuError gpuMallocHost(void** ptr, size_t bytes) { return cudaMallocHost(ptr, bytes); }
inline GpuError gpuFreeHost(void* ptr) { return cudaFreeHost(ptr); }
inline GpuError gpuStreamCreate(GpuStream* stream) { return cudaStreamCreate(stream); }
inline GpuError gpuStreamDestroy(GpuStream stream) { return cudaStreamDestroy(stream); }
inline GpuError gpuStreamSynchronize(GpuStream stream) { return cudaStreamSynchronize(stream); }
inline GpuError gpuGetMultiprocessorCount(int* count) {
  return cudaDeviceGetAttribute(count, cudaDevAttrMultiProcessorCount, 0);
}
inline GpuError gpuGetMaxThreadsPerMultiprocessor(int* count) {
  return cudaDeviceGetAttribute(count, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
}
template <typename KernelFn>
inline GpuError gpuOccupancyMaxActiveBlocksPerMultiprocessor(int* num_blocks, KernelFn kernel,
                                                             int block_size, size_t dynamic_smem) {
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(num_blocks, kernel, block_size, dynamic_smem);
}
inline GpuError gpuSetDevice(int device) { return cudaSetDevice(device); }
inline GpuError gpuGetDeviceCount(int* count) { return cudaGetDeviceCount(count); }

inline GpuError gpuEventCreate(GpuEvent* event) { return cudaEventCreate(event); }
inline GpuError gpuEventDestroy(GpuEvent event) { return cudaEventDestroy(event); }
inline GpuError gpuEventRecord(GpuEvent event) { return cudaEventRecord(event, 0); }
inline GpuError gpuEventRecordOnStream(GpuEvent event, GpuStream stream) {
  return cudaEventRecord(event, stream);
}
inline GpuError gpuEventSynchronize(GpuEvent event) { return cudaEventSynchronize(event); }
inline GpuError gpuEventElapsedTime(float* ms, GpuEvent start, GpuEvent stop) {
  return cudaEventElapsedTime(ms, start, stop);
}

#else
#error "Define USE_CUDA or USE_HIP to build GPU kernels. See kernels/common/README.md."
#endif

// Abort with a readable message if a GPU API call fails.
#define GPU_CHECK(expr)                                                              \
  do {                                                                              \
    GpuError err = (expr);                                                          \
    if (err != kGpuSuccess) {                                                       \
      std::fprintf(stderr, "GPU error %s at %s:%d\n", gpuGetErrorString(err),       \
                   __FILE__, __LINE__);                                             \
      std::exit(EXIT_FAILURE);                                                      \
    }                                                                              \
  } while (0)

// Launch a kernel and immediately check for launch errors.
#if defined(USE_HIP)
#define GPU_LAUNCH(kernel, grid, block, shmem, ...)                                 \
  do {                                                                              \
    hipLaunchKernelGGL(kernel, dim3(grid), dim3(block), (shmem), 0, __VA_ARGS__);   \
    GPU_CHECK(gpuGetLastError());                                                   \
  } while (0)
#define GPU_LAUNCH_STREAM(kernel, grid, block, shmem, stream, ...)                  \
  do {                                                                              \
    hipLaunchKernelGGL(kernel, dim3(grid), dim3(block), (shmem), (stream),           \
                       __VA_ARGS__);                                                \
    GPU_CHECK(gpuGetLastError());                                                   \
  } while (0)
#else
#define GPU_LAUNCH(kernel, grid, block, shmem, ...)                                 \
  do {                                                                              \
    kernel<<<(grid), (block), (shmem)>>>(__VA_ARGS__);                              \
    GPU_CHECK(gpuGetLastError());                                                   \
  } while (0)
#define GPU_LAUNCH_STREAM(kernel, grid, block, shmem, stream, ...)                  \
  do {                                                                              \
    kernel<<<(grid), (block), (shmem), (stream)>>>(__VA_ARGS__);                    \
    GPU_CHECK(gpuGetLastError());                                                   \
  } while (0)
#endif
