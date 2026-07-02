// race_fixed_atomic.cpp
//
// Same workload as race_demo, but the counter is a std::atomic<long long>.
// fetch_add makes each read-modify-write a single indivisible operation, so
// no update can be lost. memory_order_relaxed is enough here because we only
// need the counter itself to be correct - no other memory is published
// through it. This is the CPU analogue of CUDA/HIP atomicAdd.
#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

namespace {

constexpr int kNumThreads = 8;
constexpr int kIncrementsPerThread = 1'000'000;
constexpr int kTrials = 5;

long long runOneTrial() {
  std::atomic<long long> counter{0};

  auto worker = [&counter]() {
    for (int i = 0; i < kIncrementsPerThread; ++i) {
      // Indivisible read-modify-write; relaxed ordering is sufficient for a
      // pure counter (contrast with release/acquire when publishing data).
      counter.fetch_add(1, std::memory_order_relaxed);
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) threads.emplace_back(worker);
  for (auto& thread : threads) thread.join();
  return counter.load();
}

}  // namespace

int main() {
  constexpr long long kExpected =
      static_cast<long long>(kNumThreads) * kIncrementsPerThread;
  std::printf("race_fixed_atomic: %d threads x %d increments, expected %lld\n",
              kNumThreads, kIncrementsPerThread, kExpected);

  for (int trial = 0; trial < kTrials; ++trial) {
    const long long got = runOneTrial();
    std::printf("  trial %d: total = %10lld  %s\n", trial, got,
                got == kExpected ? "OK" : "WRONG");
  }
  std::printf("atomic fetch_add: correct without a lock.\n");
  return 0;
}
