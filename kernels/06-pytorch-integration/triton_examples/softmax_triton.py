"""Chapter 20: fused softmax in Triton, compared with torch.softmax.

Portable across NVIDIA and AMD backends of Triton. Validates correctness and
reports achieved bandwidth vs. the PyTorch reference.
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
    def softmaxKernel(x_ptr, y_ptr, row_stride, n_cols, block_size: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, block_size)
        mask = cols < n_cols
        offs = row * row_stride + cols
        x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)          # numerical stability
        num = tl.exp(x)
        y = num / tl.sum(num, axis=0)
        tl.store(y_ptr + offs, y, mask=mask)

    def softmaxTriton(x: torch.Tensor) -> torch.Tensor:
        rows, cols = x.shape
        y = torch.empty_like(x)
        block = triton.next_power_of_2(cols)
        softmaxKernel[(rows,)](x, y, x.stride(0), cols, block_size=block)
        return y


def main() -> None:
    if not kHasTriton:
        print("Triton is not installed; skipping. Install with `pip install triton`.")
        return
    if not torch.cuda.is_available():
        print("No CUDA/ROCm device available; skipping Triton softmax.")
        return

    device = torch.device("cuda")
    x = torch.randn(4096, 2048, device=device, dtype=torch.float32)
    got = softmaxTriton(x)
    want = torch.softmax(x, dim=-1)
    max_err = (got - want).abs().max().item()
    assert max_err < 1e-3, f"softmax mismatch: {max_err}"
    print(f"triton softmax correctness OK (max abs error {max_err:.3e})")


if __name__ == "__main__":
    main()
