cmake_minimum_required(VERSION 3.24)
project(gpu_kernel_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Platform auto-detection: prefer CUDA, then HIP/ROCm. The chosen platform sets
# GKLAB_PLATFORM and a compile definition (USE_CUDA / USE_HIP) that the shared
# portability header keys off of.
# ---------------------------------------------------------------------------
option(GKLAB_FORCE_HIP "Force the HIP/ROCm backend even if CUDA is present" OFF)

include(CheckLanguage)

set(GKLAB_PLATFORM "NONE")

if(NOT GKLAB_FORCE_HIP)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(GKLAB_PLATFORM "CUDA")
  endif()
endif()

if(GKLAB_PLATFORM STREQUAL "NONE")
  # CMake 3.21+ supports HIP as a first-class language.
  check_language(HIP)
  if(CMAKE_HIP_COMPILER)
    enable_language(HIP)
    set(GKLAB_PLATFORM "HIP")
  endif()
endif()

message(STATUS "gpu-kernel-lab platform: ${GKLAB_PLATFORM}")

if(GKLAB_PLATFORM STREQUAL "NONE")
  message(WARNING
    "No CUDA or HIP toolchain found. Configure will succeed but no GPU targets "
    "will be built. See kernels/common/README.md for syntax-only verification.")
endif()

set(GKLAB_COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/common)

# ---------------------------------------------------------------------------
# Helper: register a GPU executable from a single portable .cpp/.cu source.
# On CUDA we compile the .cpp as CUDA; on HIP we use hipcc via the HIP language
# or a custom compile. The same source works because of gpu_portability.h.
# ---------------------------------------------------------------------------
function(gklab_add_kernel target source)
  if(GKLAB_PLATFORM STREQUAL "CUDA")
    add_executable(${target} ${source})
    set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(${target} PRIVATE USE_CUDA)
    set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "70;80;90")
    target_include_directories(${target} PRIVATE ${GKLAB_COMMON_INCLUDE})
  elseif(GKLAB_PLATFORM STREQUAL "HIP")
    add_executable(${target} ${source})
    set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
    target_compile_definitions(${target} PRIVATE USE_HIP)
    target_include_directories(${target} PRIVATE ${GKLAB_COMMON_INCLUDE})
  else()
    # No GPU toolchain: skip building this target.
  endif()
endfunction()

# CPU parallelism foundations build on any machine, GPU toolchain or not;
# its GPU targets no-op via gklab_add_kernel when no platform was found.
add_subdirectory(00-parallel-foundations)

if(NOT GKLAB_PLATFORM STREQUAL "NONE")
  add_subdirectory(01-basics)
  add_subdirectory(02-memory)
  add_subdirectory(03-gemm)
  add_subdirectory(03-occupancy)
  add_subdirectory(04-reductions-softmax)
  add_subdirectory(05-advanced-scheduling)
  add_subdirectory(07-multi-gpu-collectives)
  add_subdirectory(exercises)
endif()
