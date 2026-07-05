"""Chapter 20: a tiled matmul in Triton, compared with torch.matmul.

A compact block-level GEMM that the Triton compiler lowers to coalesced loads
and a pipelined main loop. Validates correctness against torch.matmul.
"""

import torch

try:
    import triton
    import triton.language as tl

    kHasTriton = True
except ImportError:
    kHasTriton = False


if kHasTriton:

    @triton.jit
    def matmulKernel(a_ptr, b_ptr, c_ptr, m_size, n_size, k_size,
                     stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                     block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((block_m, block_n), dtype=tl.float32)
        for k_offset in range(0, k_size, block_k):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_size - k_offset, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_size - k_offset, other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += block_k * stride_ak
            b_ptrs += block_k * stride_bk
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)
        tl.store(c_ptrs, acc, mask=mask)

    def matmulTriton(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m_size, k_size = a.shape
        other_k_size, n_size = b.shape
        assert k_size == other_k_size
        c = torch.empty((m_size, n_size), device=a.device, dtype=torch.float32)
        block_m, block_n, block_k = 64, 64, 32
        grid = (triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))
        matmulKernel[grid](
            a, b, c, m_size, n_size, k_size,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
            block_m=block_m, block_n=block_n, block_k=block_k,
        )
        return c


def main() -> None:
    if not kHasTriton:
        print("Triton is not installed; skipping. Install with `pip install triton`.")
        return
    if not torch.cuda.is_available():
        print("No CUDA/ROCm device available; skipping Triton matmul.")
        return

    device = torch.device("cuda")
    a = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    b = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    got = matmulTriton(a, b)
    want = torch.matmul(a, b)
    max_err = (got - want).abs().max().item()
    assert max_err < 1e-1, f"matmul mismatch: {max_err}"
    print(f"triton matmul correctness OK (max abs error {max_err:.3e})")


if __name__ == "__main__":
    main()
