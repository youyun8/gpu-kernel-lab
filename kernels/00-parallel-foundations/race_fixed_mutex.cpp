// race_fixed_mutex.cpp
//
// Same workload as race_demo, but the shared counter is protected by a
// std::mutex. std::lock_guard acquires the lock for the scope of each
// increment, so read-modify-write becomes mutually exclusive and the total is
// always exact. The price is serialization: every increment takes the same
// lock, so this is also a demonstration of why fine-grained locking around a
// hot counter is slow.
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

namespace {

constexpr int kNumThreads = 8;
constexpr int kIncrementsPerThread = 1'000'000;
constexpr int kTrials = 5;

long long runOneTrial() {
  long long counter = 0;
  std::mutex counter_mutex;

  auto worker = [&counter, &counter_mutex]() {
    for (int i = 0; i < kIncrementsPerThread; ++i) {
      std::lock_guard<std::mutex> guard(counter_mutex);
      ++counter;  // exclusive access: no interleaving, no lost updates
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int t = 0; t < kNumThreads; ++t) threads.emplace_back(worker);
  for (auto& thread : threads) thread.join();
  return counter;
}

}  // namespace

int main() {
  constexpr long long kExpected =
      static_cast<long long>(kNumThreads) * kIncrementsPerThread;
  std::printf("race_fixed_mutex: %d threads x %d increments, expected %lld\n",
              kNumThreads, kIncrementsPerThread, kExpected);

  for (int trial = 0; trial < kTrials; ++trial) {
    const long long got = runOneTrial();
    std::printf("  trial %d: total = %10lld  %s\n", trial, got,
                got == kExpected ? "OK" : "WRONG");
  }
  std::printf("every trial matches -> mutual exclusion removed the race.\n");
  return 0;
}
