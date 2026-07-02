"""Chapter 17: find the kernels worth optimizing with torch.profiler.

Builds a tiny model with a memory-bound elementwise chain, profiles it, prints
the top operators by device time, and exports a Chrome trace. Run on a machine
with a CUDA or ROCm build of PyTorch.
"""

import torch
from torch.profiler import ProfilerActivity, profile


def build_inputs(device: torch.device):
    x = torch.randn(4096, 4096, device=device)
    bias = torch.randn(4096, device=device)
    return x, bias


def workload(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    # A memory-bound elementwise chain: each op reads/writes the full tensor.
    h = x + bias
    h = torch.nn.functional.gelu(h)
    return h.sum(dim=-1)


def main() -> None:
    if not torch.cuda.is_available():
        print("No CUDA/ROCm device available; this script needs a GPU build of PyTorch.")
        return

    device = torch.device("cuda")
    x, bias = build_inputs(device)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(10):
            workload(x, bias)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    prof.export_chrome_trace("trace.json")
    print("Wrote trace.json (open in chrome://tracing or https://ui.perfetto.dev/)")


if __name__ == "__main__":
    main()
