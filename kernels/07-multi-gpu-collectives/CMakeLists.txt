# Multi-GPU collectives (chapter 33). NCCL (CUDA) / RCCL (ROCm) is an optional
# dependency: when the library is missing these targets are skipped with a
# status message and the rest of the build is unaffected.

set(GKLAB_CCL_FOUND FALSE)

if(GKLAB_PLATFORM STREQUAL "CUDA")
  find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS ${NCCL_ROOT} ENV NCCL_ROOT ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    PATH_SUFFIXES include)
  find_library(NCCL_LIBRARY nccl
    HINTS ${NCCL_ROOT} ENV NCCL_ROOT
    PATH_SUFFIXES lib lib64)
  if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    set(GKLAB_CCL_FOUND TRUE)
    set(GKLAB_CCL_INCLUDE_DIR ${NCCL_INCLUDE_DIR})
    set(GKLAB_CCL_LIBRARY ${NCCL_LIBRARY})
  endif()
elseif(GKLAB_PLATFORM STREQUAL "HIP")
  # ROCm ships an rccl CMake package; fall back to a manual search if absent.
  find_package(rccl QUIET CONFIG)
  if(rccl_FOUND)
    set(GKLAB_CCL_FOUND TRUE)
    set(GKLAB_CCL_TARGET roc::rccl)
  else()
    find_path(RCCL_INCLUDE_DIR NAMES rccl/rccl.h rccl.h
      HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
      PATH_SUFFIXES include)
    find_library(RCCL_LIBRARY rccl
      HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
      PATH_SUFFIXES lib lib64)
    if(RCCL_INCLUDE_DIR AND RCCL_LIBRARY)
      set(GKLAB_CCL_FOUND TRUE)
      set(GKLAB_CCL_INCLUDE_DIR ${RCCL_INCLUDE_DIR})
      set(GKLAB_CCL_LIBRARY ${RCCL_LIBRARY})
    endif()
  endif()
endif()

if(GKLAB_CCL_FOUND)
  gklab_add_kernel(allreduce_bench allreduce_bench.cpp)
  if(TARGET allreduce_bench)
    if(DEFINED GKLAB_CCL_TARGET)
      target_link_libraries(allreduce_bench PRIVATE ${GKLAB_CCL_TARGET})
    else()
      target_include_directories(allreduce_bench PRIVATE ${GKLAB_CCL_INCLUDE_DIR})
      target_link_libraries(allreduce_bench PRIVATE ${GKLAB_CCL_LIBRARY})
    endif()
  endif()
else()
  message(STATUS "07-multi-gpu-collectives: NCCL/RCCL not found, skipping allreduce_bench")
endif()
