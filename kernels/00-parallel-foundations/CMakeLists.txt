# CPU parallelism foundations (chapter 0). These targets build without any GPU
# toolchain; OpenMP and pthread variants are skipped gracefully when the
# corresponding runtime is unavailable so the overall build never fails.

find_package(Threads QUIET)

if(Threads_FOUND)
  add_executable(race_demo race_demo.cpp)
  target_link_libraries(race_demo PRIVATE Threads::Threads)

  add_executable(race_fixed_mutex race_fixed_mutex.cpp)
  target_link_libraries(race_fixed_mutex PRIVATE Threads::Threads)

  add_executable(race_fixed_atomic race_fixed_atomic.cpp)
  target_link_libraries(race_fixed_atomic PRIVATE Threads::Threads)

  # pthread_barrier_t needs the real POSIX API, not just std::thread.
  if(CMAKE_USE_PTHREADS_INIT)
    add_executable(pthread_race pthread_race.cpp)
    target_link_libraries(pthread_race PRIVATE Threads::Threads)
  else()
    message(STATUS "00-parallel-foundations: POSIX threads not found, skipping pthread_race")
  endif()
else()
  message(STATUS "00-parallel-foundations: Threads not found, skipping CPU race demos")
endif()

find_package(OpenMP QUIET COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
  add_executable(openmp_reduction openmp_reduction.cpp)
  target_link_libraries(openmp_reduction PRIVATE OpenMP::OpenMP_CXX)
else()
  message(STATUS "00-parallel-foundations: OpenMP not found, skipping openmp_reduction")
endif()

# GPU twins of the CPU demos. gklab_add_kernel is a no-op without a GPU
# toolchain, so these lines are safe on CPU-only machines.
gklab_add_kernel(gpu_histogram_race gpu_histogram_race.cpp)
gklab_add_kernel(gpu_block_reduce gpu_block_reduce.cpp)
gklab_add_kernel(gpu_threadfence gpu_threadfence.cpp)
