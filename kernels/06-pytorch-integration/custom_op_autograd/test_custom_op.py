"""Chapter 21: register a custom op with torch.library, verify torch.compile
compatibility, and run a performance-regression check.

Uses torch.library.custom_op with a fake (meta) implementation so torch.compile
can trace it, wires up autograd, and compares timing against a recorded budget
with a tolerance.
"""

import json
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASELINE_PATH = os.path.join(_HERE, "baseline.json")


@torch.library.custom_op("gklab::gelu", mutates_args=())
def gelu(x: torch.Tensor) -> torch.Tensor:
    # Reference implementation; a real op would call a compiled kernel here.
    return torch.nn.functional.gelu(x.contiguous())


@gelu.register_fake
def _(x: torch.Tensor) -> torch.Tensor:
    # Shape/dtype propagation without running the kernel (needed by torch.compile).
    return torch.empty_like(x)


def _setup_context(ctx, inputs, output):
    (x,) = inputs
    ctx.save_for_backward(x)


def _backward(ctx, grad):
    (x,) = ctx.saved_tensors
    cdf = 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))
    pdf = 0.3989422804014327 * torch.exp(-0.5 * x * x)
    return grad * (cdf + x * pdf)


gelu.register_autograd(_backward, setup_context=_setup_context)


def benchmark_ms(fn, iters: int = 100, warmup: int = 20) -> float:
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
        print("No CUDA/ROCm device available; skipping custom-op test.")
        return

    device = torch.device("cuda")
    x = torch.randn(4096, 4096, device=device, requires_grad=True)

    # Correctness vs. reference.
    got = gelu(x)
    want = torch.nn.functional.gelu(x)
    assert (got - want).abs().max().item() < 1e-5
    print("custom op forward OK")

    # torch.compile must be able to trace the registered op without graph break.
    def model(t):
        return gelu(t).sum()

    compiled = torch.compile(model)
    compiled(x).backward()
    print("torch.compile integration OK")

    # Performance regression check against a recorded, hardware-specific budget.
    t = benchmark_ms(lambda: gelu(x))
    if os.path.exists(_BASELINE_PATH):
        with open(_BASELINE_PATH, "r", encoding="utf-8") as fh:
            baseline = json.load(fh).get("gelu_ms")
        if baseline is not None:
            assert t <= baseline * 1.10, f"perf regression: {t:.3f} ms > {baseline * 1.10:.3f} ms"
            print(f"perf OK: {t:.3f} ms (budget {baseline * 1.10:.3f} ms)")
    else:
        with open(_BASELINE_PATH, "w", encoding="utf-8") as fh:
            json.dump({"gelu_ms": t}, fh)
        print(f"recorded baseline: {t:.3f} ms -> {_BASELINE_PATH}")


if __name__ == "__main__":
    main()
