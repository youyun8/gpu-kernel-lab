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

__global__ void runTasksStatic(const int* costs, unsigned int* out, int num_tasks) {
  int task = blockIdx.x;
  if (task >= num_tasks) return;
  unsigned int total = runTask(task, costs[task]);
  if (threadIdx.x == 0) out[task] = total;
}

__global__ void runTasksPersistent(const int* costs, unsigned int* out, int* next_task, int num_tasks) {
  // The whole block must agree on which chunk it owns. Only thread 0 performs
  // the atomic dequeue; the result is broadcast through shared memory so every
  // thread takes the same control-flow path through the block-wide barriers in
  // runTask()/blockReduceSum(). Letting every thread call atomicAdd() would give
  // each thread a different `begin`, diverge them across the loop, and corrupt
  // the __syncthreads()-based reduction (and can hang the block).
  __shared__ int begin_shared;
  while (true) {
    if (threadIdx.x == 0) begin_shared = atomicAdd(next_task, kChunkSize);
    __syncthreads();
    int begin = begin_shared;
    if (begin >= num_tasks) break;
    int end = min(begin + kChunkSize, num_tasks);
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

  int sm_count = 1;
  GPU_CHECK(gpuGetMultiprocessorCount(&sm_count));
  const int persistent_blocks = std::max(1, sm_count * 2);

  int* dev_costs = nullptr;
  int* dev_next = nullptr;
  unsigned int* dev_out = nullptr;
  const size_t cost_bytes = costs.size() * sizeof(int);
  const size_t out_bytes = reference.size() * sizeof(unsigned int);
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_costs), cost_bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_out), out_bytes));
  GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&dev_next), sizeof(int)));
  GPU_CHECK(gpuMemcpyHostToDevice(dev_costs, costs.data(), cost_bytes));

  const size_t bytes_moved = cost_bytes + out_bytes;
  constexpr double kPeakGbPerSec = 1555.0;

  auto runStatic = [&]() { GPU_LAUNCH(runTasksStatic, kTasks, kBlockSize, 0, dev_costs, dev_out, kTasks); };
  runStatic();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_out, out_bytes));
  if (!verifyTasks("tasks_static", result, reference)) return EXIT_FAILURE;
  gklab::report("tasks_static", gklab::benchmarkKernel(runStatic, bytes_moved, 0.0), kPeakGbPerSec, 0.0);

  auto runPersistent = [&]() {
    GPU_CHECK(gpuMemset(dev_next, 0, sizeof(int)));
    GPU_LAUNCH(runTasksPersistent, persistent_blocks, kBlockSize, 0, dev_costs, dev_out, dev_next, kTasks);
  };
  runPersistent();
  GPU_CHECK(gpuDeviceSynchronize());
  GPU_CHECK(gpuMemcpyDeviceToHost(result.data(), dev_out, out_bytes));
  if (!verifyTasks("tasks_persistent", result, reference)) return EXIT_FAILURE;
  gklab::report("tasks_persistent", gklab::benchmarkKernel(runPersistent, bytes_moved, 0.0), kPeakGbPerSec,
                0.0);

  std::printf("persistent blocks: %d (SM/CU count %d x 2)\n", persistent_blocks, sm_count);

  GPU_CHECK(gpuFree(dev_costs));
  GPU_CHECK(gpuFree(dev_out));
  GPU_CHECK(gpuFree(dev_next));
  return EXIT_SUCCESS;
}
