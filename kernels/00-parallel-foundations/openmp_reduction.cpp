// openmp_reduction.cpp
//
// The same shared-counter workload expressed three ways in OpenMP:
//   1. broken:    #pragma omp parallel for with an unsynchronized counter
//   2. critical:  correct but serialized on every increment
//   3. reduction: correct and fast - each thread accumulates privately and
//                 OpenMP combines the partials at the end
// The reduction clause is the CPU twin of a GPU block/warp reduction: privatize,
// then combine, instead of hammering one shared address.
#include <cstdio>
#include <omp.h>

namespace {

constexpr int kNumIterations = 8'000'000;

long long runRacy() {
  long long counter = 0;
#pragma omp parallel for
  for (int i = 0; i < kNumIterations; ++i) {
    ++counter;  // RACE: every thread hits the same shared variable
  }
  return counter;
}

long long runCritical() {
  long long counter = 0;
#pragma omp parallel for
  for (int i = 0; i < kNumIterations; ++i) {
#pragma omp critical
    ++counter;  // correct, but one giant lock -> serialized and slow
  }
  return counter;
}

long long runReduction() {
  long long counter = 0;
#pragma omp parallel for reduction(+ : counter)
  for (int i = 0; i < kNumIterations; ++i) {
    ++counter;  // each thread increments a private copy; combined at the end
  }
  return counter;
}

}  // namespace

int main() {
  constexpr long long kExpected = kNumIterations;
  std::printf("openmp_reduction: %d iterations on %d threads, expected %lld\n",
              kNumIterations, omp_get_max_threads(), kExpected);

  const double t0 = omp_get_wtime();
  const long long racy = runRacy();
  const double t1 = omp_get_wtime();
  const long long critical = runCritical();
  const double t2 = omp_get_wtime();
  const long long reduced = runReduction();
  const double t3 = omp_get_wtime();

  std::printf("  racy      : total = %10lld  %-5s  %8.1f ms\n", racy,
              racy == kExpected ? "OK?" : "WRONG", (t1 - t0) * 1e3);
  std::printf("  critical  : total = %10lld  %-5s  %8.1f ms\n", critical,
              critical == kExpected ? "OK" : "WRONG", (t2 - t1) * 1e3);
  std::printf("  reduction : total = %10lld  %-5s  %8.1f ms\n", reduced,
              reduced == kExpected ? "OK" : "WRONG", (t3 - t2) * 1e3);
  std::printf("reduction is both correct and much faster than critical.\n");
  return 0;
}
