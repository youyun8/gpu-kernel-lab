// race_demo.cpp
//
// Deliberately broken: several std::thread workers increment one shared
// counter with a plain, unsynchronized read-modify-write. Two threads that
// both read the same old value write back the same new value, so one of the
// two increments is lost ("lost update"). Run it a few times: the total is
// below the expected value and changes from run to run.
//
// This program has undefined behavior by design (a data race); it exists so
// the fixed versions (race_fixed_mutex / race_fixed_atomic) have a baseline.
#include <cstdio>
#include <thread>
#include <vector>

namespace {

constexpr int kNumThreads = 8;
constexpr int kIncrementsPerThread = 1'000'000;
constexpr int kTrials = 5;

long long runOneTrial() {
  // `volatile` forces every ++ to be a real load-add-store to memory instead of
  // a register accumulation the optimizer would otherwise hoist out of the loop.
  // It does NOT make the access atomic, so the data race stays intact and stays
  // observable even with -O2: two threads that read the same old value both
  // write back the same new value, and one increment is lost.
  volatile long long counter = 0;

  auto worker = [&counter]() {
    for (int i = 0; i < kIncrementsPerThread; ++i) {
      ++counter;  // RACE: unsynchronized read-modify-write on shared memory
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
  std::printf("race_demo: %d threads x %d increments, expected total %lld\n",
              kNumThreads, kIncrementsPerThread, kExpected);

  int wrong_trials = 0;
  for (int trial = 0; trial < kTrials; ++trial) {
    const long long got = runOneTrial();
    const double lost_pct = 100.0 * static_cast<double>(kExpected - got) /
                            static_cast<double>(kExpected);
    if (got != kExpected) ++wrong_trials;
    std::printf("  trial %d: total = %10lld  (lost %5.1f%% of updates) %s\n",
                trial, got, lost_pct, got == kExpected ? "" : "WRONG");
  }

  if (wrong_trials > 0) {
    std::printf(
        "%d/%d trials lost updates -> the unsynchronized counter has a data "
        "race. Compare with race_fixed_mutex / race_fixed_atomic.\n",
        wrong_trials, kTrials);
  } else {
    // Only reachable with no real parallelism (e.g. a single core): the race
    // exists but never got a chance to interleave.
    std::printf(
        "all trials happened to match; the race did not interleave here (too "
        "few cores?), but the code is still racy by construction.\n");
  }
  return 0;
}
