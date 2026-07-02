// ccl_portability.h
//
// Thin portability layer over NCCL (CUDA) and RCCL (ROCm) so the same
// collective-communication example compiles on both platforms. RCCL ships an
// NCCL-compatible API - identical ncclXxx names and semantics - so unlike
// gpu_portability.h no name mapping is needed here; only the header location
// and the link library differ (handled by the build system).
//
// Kept separate from gpu_portability.h because NCCL/RCCL is an optional
// dependency: kernels that do not use collectives must not require it.
#pragma once

#include <cstdio>
#include <cstdlib>

#include "gpu_portability.h"

#if defined(USE_HIP)
// RCCL installs rccl.h either at the include root (older ROCm) or under
// rccl/ (newer ROCm); accept both.
#if __has_include(<rccl/rccl.h>)
#include <rccl/rccl.h>
#else
#include <rccl.h>
#endif
#elif defined(USE_CUDA)
#include <nccl.h>
#else
#error "Define USE_CUDA or USE_HIP to build collective examples."
#endif

// Abort with a readable message if a NCCL/RCCL call fails.
#define CCL_CHECK(expr)                                                             \
  do {                                                                              \
    ncclResult_t ccl_err = (expr);                                                  \
    if (ccl_err != ncclSuccess) {                                                   \
      std::fprintf(stderr, "NCCL/RCCL error %s at %s:%d\n",                         \
                   ncclGetErrorString(ccl_err), __FILE__, __LINE__);                \
      std::exit(EXIT_FAILURE);                                                      \
    }                                                                              \
  } while (0)
