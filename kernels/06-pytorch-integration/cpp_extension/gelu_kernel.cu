// gelu_kernel.cu
//
// GELU forward and backward CUDA/HIP kernels for a PyTorch cpp_extension.
// PyTorch's build system hipifies .cu sources automatically for ROCm.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int kBlockSize = 256;
constexpr float kSqrt2Inv = 0.7071067811865476f;   // 1/sqrt(2)
constexpr float kSqrt2OverPi = 0.7978845608028654f;  // sqrt(2/pi) for tanh grad

__global__ void geluForwardKernel(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float v = x[i];
    y[i] = 0.5f * v * (1.0f + erff(v * kSqrt2Inv));  // exact GELU
  }
}

// Backward using the exact-GELU derivative:
//   d/dx [0.5 x (1 + erf(x/sqrt2))]
//     = 0.5 (1 + erf(x/sqrt2)) + x * (1/sqrt(2pi)) * exp(-x^2/2)
__global__ void geluBackwardKernel(const float* grad, const float* x, float* dx, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float v = x[i];
    const float cdf = 0.5f * (1.0f + erff(v * kSqrt2Inv));
    const float pdf = kSqrt2OverPi * 0.5f * expf(-0.5f * v * v);  // (1/sqrt(2pi)) exp(-x^2/2)
    dx[i] = grad[i] * (cdf + v * pdf);
  }
}

}  // namespace

torch::Tensor geluForwardCuda(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA/HIP tensor");
  x = x.contiguous();
  TORCH_CHECK(x.scalar_type() == at::kFloat, "x must be float32");
  const at::cuda::CUDAGuard guard(x.device());

  auto y = torch::empty_like(x);
  const int n = static_cast<int>(x.numel());
  const int grid = (n + kBlockSize - 1) / kBlockSize;
  auto stream = at::cuda::getCurrentCUDAStream();
  geluForwardKernel<<<grid, kBlockSize, 0, stream>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
  return y;
}

torch::Tensor geluBackwardCuda(torch::Tensor grad, torch::Tensor x) {
  TORCH_CHECK(grad.is_cuda() && x.is_cuda(), "tensors must be on device");
  grad = grad.contiguous();
  x = x.contiguous();
  const at::cuda::CUDAGuard guard(x.device());

  auto dx = torch::empty_like(x);
  const int n = static_cast<int>(x.numel());
  const int grid = (n + kBlockSize - 1) / kBlockSize;
  auto stream = at::cuda::getCurrentCUDAStream();
  geluBackwardKernel<<<grid, kBlockSize, 0, stream>>>(
      grad.data_ptr<float>(), x.data_ptr<float>(), dx.data_ptr<float>(), n);
  return dx;
}
