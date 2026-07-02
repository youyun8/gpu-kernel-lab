"""Chapter 20: a tiled matmul in Triton, compared with torch.matmul.

A compact block-level GEMM that the Triton compiler lowers to coalesced loads
and a pipelined main loop. Validates correctness against torch.matmul.
"""

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                       stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=mask)

    def matmul_triton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        M, K = a.shape
        K2, N = b.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)
        block_m, block_n, block_k = 64, 64, 32
        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
        _matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        )
        return c


def main() -> None:
    if not _HAS_TRITON:
        print("Triton is not installed; skipping. Install with `pip install triton`.")
        return
    if not torch.cuda.is_available():
        print("No CUDA/ROCm device available; skipping Triton matmul.")
        return

    device = torch.device("cuda")
    a = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    b = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    got = matmul_triton(a, b)
    want = torch.matmul(a, b)
    max_err = (got - want).abs().max().item()
    assert max_err < 1e-1, f"matmul mismatch: {max_err}"
    print(f"triton matmul correctness OK (max abs error {max_err:.3e})")


if __name__ == "__main__":
    main()
