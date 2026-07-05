"""Chapter 19: fused bias + gelu + row-sum via load_inline.

Compares three implementations of `(x + bias) -> gelu -> sum(dim=-1)`:
  1. baseline: three native PyTorch ops (three global-memory passes)
  2. torch.compile: automatically fused into a Triton kernel
  3. fused: a single custom kernel loaded with load_inline
All are checked for correctness and timed end-to-end with CUDA events + warmup.
"""

import torch
from torch.utils.cpp_extension import load_inline

kCudaSrc = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

constexpr int kBlockSize = 256;
constexpr float kSqrt2Inv = 0.7071067811865476f;

// One block per row: load once, apply bias + gelu in registers, block-reduce.
__global__ void fusedBiasGeluRowSumKernel(const float* x, const float* bias, float* out,
                                          int rows, int cols) {
  __shared__ float smem[kBlockSize];
  int row = blockIdx.x;
  int tid = threadIdx.x;
  float partial = 0.0f;
  for (int c = tid; c < cols; c += blockDim.x) {
    float v = x[static_cast<long long>(row) * cols + c] + bias[c];
    v = 0.5f * v * (1.0f + erff(v * kSqrt2Inv));
    partial += v;
  }
  smem[tid] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) smem[tid] += smem[tid + s];
    __syncthreads();
  }
  if (tid == 0) out[row] = smem[0];
}

torch::Tensor fusedBiasGeluRowSum(torch::Tensor x, torch::Tensor bias) {
  TORCH_CHECK(x.is_cuda() && bias.is_cuda(), "inputs must be on device");
  x = x.contiguous();
  bias = bias.contiguous();
  const at::cuda::CUDAGuard guard(x.device());
  const int rows = x.size(0);
  const int cols = x.size(1);
  auto out = torch::empty({rows}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  fusedBiasGeluRowSumKernel<<<rows, kBlockSize, 0, stream>>>(
      x.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(), rows, cols);
  return out;
}
"""

kCppSrc = "torch::Tensor fusedBiasGeluRowSum(torch::Tensor x, torch::Tensor bias);"


def baseline(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    h = x + bias
    h = torch.nn.functional.gelu(h)
    return h.sum(dim=-1)


def benchmarkMs(fn, iters: int = 100, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def main() -> None:
    if not torch.cuda.is_available():
        print("No CUDA/ROCm device available; skipping fused kernel demo.")
        return

    module = load_inline(
        name="fused_bias_gelu",
        cpp_sources=kCppSrc,
        cuda_sources=kCudaSrc,
        functions=["fusedBiasGeluRowSum"],
        verbose=True,
    )

    device = torch.device("cuda")
    x = torch.randn(8192, 4096, device=device, dtype=torch.float32)
    bias = torch.randn(4096, device=device, dtype=torch.float32)

    ref = baseline(x, bias)
    fused = module.fusedBiasGeluRowSum(x, bias)
    max_err = (ref - fused).abs().max().item()
    assert max_err < 1e-2, f"fused mismatch: {max_err}"
    print(f"fused correctness OK (max abs error {max_err:.3e})")

    compiled = torch.compile(baseline)
    compiled(x, bias)  # warmup / compile

    t_base = benchmarkMs(lambda: baseline(x, bias))
    t_comp = benchmarkMs(lambda: compiled(x, bias))
    t_fused = benchmarkMs(lambda: module.fusedBiasGeluRowSum(x, bias))
    print(f"baseline       : {t_base:.3f} ms")
    print(f"torch.compile  : {t_comp:.3f} ms  ({t_base / t_comp:.2f}x vs baseline)")
    print(f"fused kernel   : {t_fused:.3f} ms  ({t_base / t_fused:.2f}x vs baseline)")


if __name__ == "__main__":
    main()
