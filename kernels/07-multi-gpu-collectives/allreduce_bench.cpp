// allreduce_bench.cpp
//
// Single-process, multi-device all-reduce benchmark in the style of
// nccl-tests: one ncclComm per visible GPU (ncclCommInitAll), a sweep over
// message sizes, correctness check against the analytic expected value, and
// the two standard bandwidth numbers per size:
//
//   algbw = message_size / time
//       "application" bandwidth: how fast the caller sees its buffer reduced.
//   busbw = algbw * 2*(n-1)/n         (for all-reduce)
//       "bus" bandwidth: the traffic each GPU actually puts on the wire.
//       Ring all-reduce moves every element off/onto each GPU 2*(n-1)/n
//       times (n-1 reduce-scatter steps + n-1 all-gather steps, each moving
//       size/n per GPU), so busbw is comparable to the link's peak bandwidth
//       regardless of GPU count.
//
// Requires >= 2 GPUs and NCCL (CUDA) or RCCL (ROCm) at build time.
#include <chrono>
#include <cstdio>
#include <vector>

#include "ccl_portability.h"
#include "gpu_portability.h"

namespace {

constexpr int kWarmupIters = 5;
constexpr int kTimedIters = 20;
constexpr size_t kMinBytes = 1 << 20;   // 1 MiB
constexpr size_t kMaxBytes = 1 << 28;   // 256 MiB

double allreduceOnce(const std::vector<ncclComm_t>& comms, const std::vector<GpuStream>& streams,
                     const std::vector<float*>& send_bufs, const std::vector<float*>& recv_bufs,
                     size_t count, int iters) {
  const int num_gpus = static_cast<int>(comms.size());
  const auto start = std::chrono::steady_clock::now();
  for (int it = 0; it < iters; ++it) {
    // Group the per-device calls so NCCL launches them as one collective.
    CCL_CHECK(ncclGroupStart());
    for (int g = 0; g < num_gpus; ++g) {
      CCL_CHECK(ncclAllReduce(send_bufs[g], recv_bufs[g], count, ncclFloat, ncclSum,
                              comms[g], streams[g]));
    }
    CCL_CHECK(ncclGroupEnd());
  }
  for (int g = 0; g < num_gpus; ++g) {
    GPU_CHECK(gpuSetDevice(g));
    GPU_CHECK(gpuStreamSynchronize(streams[g]));
  }
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(stop - start).count() / iters;
}

}  // namespace

int main() {
  int num_gpus = 0;
  GPU_CHECK(gpuGetDeviceCount(&num_gpus));
  if (num_gpus < 2) {
    std::fprintf(stderr, "allreduce_bench: needs >= 2 GPUs, found %d\n", num_gpus);
    return EXIT_FAILURE;
  }
  std::printf("allreduce_bench: %d GPUs, float sum, %d timed iters per size\n", num_gpus,
              kTimedIters);

  // One communicator per device in this single process.
  std::vector<ncclComm_t> comms(num_gpus);
  CCL_CHECK(ncclCommInitAll(comms.data(), num_gpus, nullptr));

  const size_t max_count = kMaxBytes / sizeof(float);
  std::vector<GpuStream> streams(num_gpus);
  std::vector<float*> send_bufs(num_gpus);
  std::vector<float*> recv_bufs(num_gpus);
  std::vector<float> host(max_count);

  for (int g = 0; g < num_gpus; ++g) {
    GPU_CHECK(gpuSetDevice(g));
    GPU_CHECK(gpuStreamCreate(&streams[g]));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&send_bufs[g]), kMaxBytes));
    GPU_CHECK(gpuMalloc(reinterpret_cast<void**>(&recv_bufs[g]), kMaxBytes));
    // Rank-specific pattern so the reduced value is verifiable: element i on
    // rank g is (g + 1), so the all-reduced value is n*(n+1)/2 everywhere.
    for (size_t i = 0; i < max_count; ++i) host[i] = static_cast<float>(g + 1);
    GPU_CHECK(gpuMemcpyHostToDevice(send_bufs[g], host.data(), kMaxBytes));
  }
  const float expected = static_cast<float>(num_gpus) * (num_gpus + 1) / 2.0f;

  std::printf("%12s %12s %12s %12s\n", "size(B)", "time(us)", "algbw(GB/s)", "busbw(GB/s)");
  for (size_t bytes = kMinBytes; bytes <= kMaxBytes; bytes <<= 1) {
    const size_t count = bytes / sizeof(float);

    allreduceOnce(comms, streams, send_bufs, recv_bufs, count, kWarmupIters);

    // Correctness: every element on every rank must equal sum(1..n).
    for (int g = 0; g < num_gpus; ++g) {
      GPU_CHECK(gpuSetDevice(g));
      GPU_CHECK(gpuMemcpyDeviceToHost(host.data(), recv_bufs[g], bytes));
      for (size_t i = 0; i < count; ++i) {
        if (host[i] != expected) {
          std::fprintf(stderr, "verification FAILED: rank %d elem %zu got %f want %f\n", g, i,
                       host[i], expected);
          return EXIT_FAILURE;
        }
      }
    }

    const double seconds = allreduceOnce(comms, streams, send_bufs, recv_bufs, count, kTimedIters);
    const double algbw = static_cast<double>(bytes) / seconds / 1.0e9;
    const double busbw = algbw * 2.0 * (num_gpus - 1) / num_gpus;
    std::printf("%12zu %12.1f %12.2f %12.2f\n", bytes, seconds * 1.0e6, algbw, busbw);
  }
  std::printf("correctness OK on all ranks (expected %.1f)\n", expected);

  for (int g = 0; g < num_gpus; ++g) {
    GPU_CHECK(gpuSetDevice(g));
    GPU_CHECK(gpuFree(send_bufs[g]));
    GPU_CHECK(gpuFree(recv_bufs[g]));
    GPU_CHECK(gpuStreamDestroy(streams[g]));
    CCL_CHECK(ncclCommDestroy(comms[g]));
  }
  return EXIT_SUCCESS;
}
