# Occupancy tuning levers (chapter b8). One runnable experiment that queries
# achieved occupancy per block size and times a high-occupancy vs low-occupancy
# + high-ILP configuration of the same workload.
gklab_add_kernel(tuning_levers_demo tuning_levers_demo.cpp)
