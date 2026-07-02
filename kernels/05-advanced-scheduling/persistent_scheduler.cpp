// persistent_scheduler.cpp
//
// Chapter 27 reference solution: compare a static one-block-per-task schedule
// with a persistent work-queue scheduler on skewed task costs.
#include <algorithm>
#include <cstdio>
#include <vector>

#include "benchmark.h"
#include "gpu_portability.h"

namespace {

constexpr int kBlockSize = 256;
constexpr int kChunkSize = 4;

__device__ unsigned int blockReduceSum(unsigned int value) {
  __shared__ unsigned int smem[kBlockSize];
  int tid = threadIdx.x;
  smem[tid] = value;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
  }
  return smem[0];
}

__device__ unsigned int runTask(int task, int iters) {
  unsigned int local = 0;
  for (int i = threadIdx.x; i < iters; i += blockDim.x) {
    local += static_cast<unsigned int>((i * 17 + task * 13) & 255);
  }
  return blockReduceSum(local);
}

__global__ void runTasksStatic(const int* costs, unsigned int* out, int numTasks) {
  int task = blockIdx.x;
  if (task >= numTasks) return;
  unsigned int total = runTask(task, costs[task]);
  if (threadIdx.x == 0) out[task] = total;
}

__global__ void runTasksPersistent(const int* costs, unsigned int* out, int* nextTask, int numTasks) {
  while (true) {
    int begin = atomicAdd(nextTask, kChunkSize);
    if (begin >= numTasks) break;
    int end = min(begin + kChunkSize, numTasks);
    for (int task = begin; task < end; ++task) {
      unsigned int total = runTask(task, costs[task]);
      if (threadIdx.x == 0) out[task] = total;
      __syncthreads();
    }
  }
}

unsigned int cpuTask(int task, int iters) {
  unsigned int total = 0;
  for (int i = 0; i < iters; ++i) total += static_cast<unsigned int>((i * 17 + task * 13) & 255);
  return total;
}

bool verifyTasks(const char* name, const std::vector<unsigned int>& got,
                 const std::vector<unsigned int>& want) {
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != want[i]) {
      std::fprintf(stderr, "%s FAILED: task %zu got %u want %u\n", name, i, got[i], want[i]);
      return false;
    }
  }
  std::printf("%s correctness OK\n", name);
  return true;
}

}  // namespace

int main() {
  constexpr int kTasks = 4096;
  std::vector<int> costs(kTasks);
  std::vector<unsigned int> reference(kTasks);
  std::vector<unsigned int> result(kTasks);

  for (int task = 0; task < kTasks; ++task) {
    int cost = 512 + ((task * 37) & 2047);
    if (task % 97 == 0) cost *= 32;
    if (task % 389 == 0) cost *= 64;
    costs[task] = cost;
    reference[task] = cpuTask(task, cost);
  }

  int smCount = 1;
  GPU_CHECK(gpuGetMultiprocessorCount(&smCount));
  const int persistentBlocks = std::max(1, smCount * 2);

  int* devCosts = nullptr;
  int* devNext = nullptr;
  unsigned int* devOut = nullptr;
  const size_t costBytes = costs.size() * sizeof(int);
  const size_t outBytes = reference.size() * sizeof(unsigned int);
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devCosts), costBytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devOut), outBytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&devNext), sizeof(int)));
  GPU_CHECK(gpuMemcpyHostToDevice(devCosts, costs.data(), costBytes));

  const size_t bytesMoved = costBytes + outBytes;
  constexpr double kPeakGbPerSec = 1555.0;

  auto runStatic = [&]() { GPU_LAUNCH(runTasksStatic, kTasks, kBlockSize, 0, devCosts, devOut, kTasks); };
  runStatic();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devOut, outBytes));
  if (!verifyTasks("tasks_static", result, reference)) return EXIT_FAILURE;
  gklab::report("tasks_static", gklab::benchmarkKernel(runStatic, bytesMoved, 0.0), kPeakGbPerSec, 0.0);

  auto runPersistent = [&]() {
    GPU_CHECK(gpuMemset(devNext, 0, sizeof(int)));
    GPU_LAUNCH(runTasksPersistent, persistentBlocks, kBlockSize, 0, devCosts, devOut, devNext, kTasks);
  };
  runPersistent();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), devOut, outBytes));
  if (!verifyTasks("tasks_persistent", result, reference)) return EXIT_FAILURE;
  gklab::report("tasks_persistent", gklab::benchmarkKernel(runPersistent, bytesMoved, 0.0), kPeakGbPerSec,
                0.0);

  std::printf("persistent blocks: %d (SM/CU count %d x 2)\n", persistentBlocks, smCount);

  GPU_CHECK(gpuFree(devCosts));
  GPU_CHECK(gpuFree(devOut));
  GPU_CHECK(gpuFree(devNext));
  return EXIT_SUCCESS;
}
