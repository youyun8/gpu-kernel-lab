gklab_add_kernel(bandwidth_probe bandwidth_probe.cpp)
gklab_add_kernel(stride_bandwidth stride_bandwidth.cpp)
gklab_add_kernel(roofline_probe roofline_probe.cpp)
gklab_add_kernel(transpose transpose.cpp)
gklab_add_kernel(vectorized_copy vectorized_copy.cpp)
gklab_add_kernel(saxpy_unroll saxpy_unroll.cpp)

add_subdirectory(access-patterns)
