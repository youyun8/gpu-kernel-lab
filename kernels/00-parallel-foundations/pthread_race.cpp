// pthread_race.cpp
//
// The lost-update demo written against the raw POSIX threads API, for readers
// coming from C: pthread_create/pthread_join, pthread_mutex_t for the fix,
// and a pthread_barrier_t so all workers start the hot loop at the same time
// (maximizing the interleaving window in the racy variant).
#include <cstdio>
#include <pthread.h>

namespace {

constexpr int kNumThreads = 8;
constexpr int kIncrementsPerThread = 1'000'000;

struct WorkerArgs {
  long long* counter;
  pthread_mutex_t* mutex;  // nullptr -> racy variant
  pthread_barrier_t* start_barrier;
};

void* worker(void* raw_args) {
  WorkerArgs* args = static_cast<WorkerArgs*>(raw_args);
  // Wait until every thread is created so the loops really overlap.
  pthread_barrier_wait(args->start_barrier);
  for (int i = 0; i < kIncrementsPerThread; ++i) {
    if (args->mutex != nullptr) {
      pthread_mutex_lock(args->mutex);
      ++*args->counter;
      pthread_mutex_unlock(args->mutex);
    } else {
      ++*args->counter;  // RACE: unsynchronized read-modify-write
    }
  }
  return nullptr;
}

long long runTrial(bool use_mutex) {
  long long counter = 0;
  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_barrier_t start_barrier;
  pthread_barrier_init(&start_barrier, nullptr, kNumThreads);

  WorkerArgs args{&counter, use_mutex ? &mutex : nullptr, &start_barrier};
  pthread_t threads[kNumThreads];
  for (int t = 0; t < kNumThreads; ++t) {
    pthread_create(&threads[t], nullptr, worker, &args);
  }
  for (int t = 0; t < kNumThreads; ++t) pthread_join(threads[t], nullptr);

  pthread_barrier_destroy(&start_barrier);
  pthread_mutex_destroy(&mutex);
  return counter;
}

}  // namespace

int main() {
  constexpr long long kExpected =
      static_cast<long long>(kNumThreads) * kIncrementsPerThread;
  std::printf("pthread_race: %d threads x %d increments, expected %lld\n",
              kNumThreads, kIncrementsPerThread, kExpected);

  for (int trial = 0; trial < 3; ++trial) {
    std::printf("  racy  trial %d: total = %10lld\n", trial, runTrial(false));
  }
  for (int trial = 0; trial < 3; ++trial) {
    const long long got = runTrial(true);
    std::printf("  mutex trial %d: total = %10lld  %s\n", trial, got,
                got == kExpected ? "OK" : "WRONG");
  }
  return 0;
}
