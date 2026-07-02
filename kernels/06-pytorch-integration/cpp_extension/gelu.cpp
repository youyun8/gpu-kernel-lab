// gelu.cpp
//
// Host-side bindings for the GELU extension. The kernels live in gelu_kernel.cu.
#include <torch/extension.h>

torch::Tensor geluForwardCuda(torch::Tensor x);
torch::Tensor geluBackwardCuda(torch::Tensor grad, torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gelu_forward", &geluForwardCuda, "GELU forward (CUDA/HIP)");
  m.def("gelu_backward", &geluBackwardCuda, "GELU backward (CUDA/HIP)");
}
