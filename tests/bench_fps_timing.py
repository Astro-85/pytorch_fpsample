"""Benchmark FPS timing on CPU vs GPU.

Runs bucket-based farthest point sampling with configurable N, D, and K.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch

from torch_quickfps import sample_idx, sample_idx_baseline


def _make_gaussian_blobs(
    batch_size: int,
    n: int,
    d: int,
    num_blobs: int = 8,
    cluster_std: float = 0.5,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create high-dimensional Gaussian blobs with shape [B, N, D]."""
    centers = torch.randn(batch_size, num_blobs, d, device=device, dtype=dtype) * 5.0
    assignments = torch.randint(
        0,
        num_blobs,
        (batch_size, n),
        device=device,
        dtype=torch.long,
    )
    chosen = centers.gather(
        1,
        assignments.unsqueeze(-1).expand(-1, -1, d),
    )
    noise = torch.randn(batch_size, n, d, device=device, dtype=dtype) * cluster_std
    return chosen + noise


def _time_fps(
    fn,
    x: torch.Tensor,
    k: int,
    warmup: int,
    iters: int,
    h: int,
    low_d: int,
    start_idx: int,
) -> float:
    """Return average time per iteration in milliseconds."""
    # Warm-up
    for _ in range(warmup):
        _ = fn(x, k, h=h, low_d=low_d, start_idx=start_idx)
    if x.is_cuda:
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = fn(x, k, h=h, low_d=low_d, start_idx=start_idx)
    if x.is_cuda:
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000.0 / iters


def _coverage_metric(
    x: torch.Tensor,
    idx: torch.Tensor,
    eval_points: int = 1024,
    seed: int = 0,
) -> tuple[float, float]:
    """Return (mean, max) min-distance to sampled points.

    Uses a subset of points for efficiency.
    """
    with torch.no_grad():
        b, n, d = x.shape
        m = min(eval_points, n)
        if m <= 0:
            return float("nan"), float("nan")
        gen = torch.Generator(device=x.device)
        gen.manual_seed(seed)
        perm = torch.randperm(n, generator=gen, device=x.device)[:m]
        x_eval = x.index_select(1, perm).float()
        x_sampled = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, d)).float()
        dists = torch.cdist(x_eval, x_sampled, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        min_d = dists.min(dim=2).values
        return min_d.mean().item(), min_d.max().item()


def _format_table(rows: List[Dict[str, Optional[float]]]) -> str:
    headers = [
        "N",
        "D",
        "K",
        "MeanCPU",
        "MeanBase",
        "MeanGPU",
        "CPU ms",
        "GPU base ms",
        "GPU ms",
        "CPU/GPU",
        "Base/GPU",
    ]
    lines = [
        f"{headers[0]:>6} {headers[1]:>6} {headers[2]:>6} "
        f"{headers[3]:>10} {headers[4]:>10} {headers[5]:>10} {headers[6]:>10} "
        f"{headers[7]:>12} {headers[8]:>10} {headers[9]:>10} {headers[10]:>10}"
    ]
    lines.append("-" * len(lines[0]))
    for row in rows:
        mean_cpu = row["mean_cpu"]
        mean_base = row["mean_base"]
        mean_gpu = row["mean_gpu"]
        cpu_ms = row["cpu_ms"]
        gpu_base_ms = row["gpu_base_ms"]
        gpu_ms = row["gpu_ms"]
        speedup_cpu = row["speedup_cpu"]
        speedup_base = row["speedup_base"]
        lines.append(
            f"{row['n']:>6} {row['d']:>6} {row['k']:>6} "
            f"{mean_cpu:>10.3e} "
            f"{mean_base:>10.3e} "
            f"{mean_gpu:>10.3e} "
            f"{cpu_ms:>10.3f} "
            f"{gpu_base_ms:>12.3f} "
            f"{gpu_ms:>10.3f} "
            f"{speedup_cpu:>10.2f}x"
            f"{speedup_base:>10.2f}x"
        )
    return "\n".join(lines)


def main() -> None:
    torch.manual_seed(0)

    batch_size = 8
    warmup = 5
    iters = 5
    verbose = True
    
    cpu_h = 3
    gpu_h = 8
    low_d = 8

    sequence_lengths = [1000, 2000, 5000, 10000]
    dims = [8, 64, 256, 1024, 4096]
    dtypes = [torch.float32, torch.float16]

    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        raise RuntimeError("CUDA is not available. GPU vs CPU timing requires a CUDA GPU.")

    for dtype in dtypes:
        if verbose:
            print(f"\n=== DTYPE: {dtype} ===")
        results: List[Dict[str, Optional[float]]] = []

        for n in sequence_lengths:
            k = n // 4
            for d in dims:
                if verbose:
                    print(
                        f"Testing N={n}, D={d}, K={k}, batch={batch_size}, "
                        f"warmup={warmup}, iters={iters}"
                    )
                x_cpu = _make_gaussian_blobs(batch_size, n, d, device="cpu", dtype=dtype)
                start_idx = 0
                if verbose:
                    print("  -> CPU timing...")
                cpu_ms = _time_fps(
                    sample_idx, x_cpu, k, warmup, iters, h=cpu_h, low_d=low_d, start_idx=start_idx
                )

                x_gpu = x_cpu.to("cuda", non_blocking=True)
                if verbose:
                    print("  -> GPU timing...")
                gpu_ms = _time_fps(
                    sample_idx, x_gpu, k, warmup, iters, h=gpu_h, low_d=low_d, start_idx=start_idx
                )
                if verbose:
                    print("  -> GPU baseline timing...")
                gpu_base_ms = _time_fps(
                    sample_idx_baseline,
                    x_gpu,
                    k,
                    warmup,
                    iters,
                    h=gpu_h,
                    low_d=low_d,
                    start_idx=start_idx,
                )

                idx_cpu = sample_idx(
                    x_cpu, k, h=cpu_h, low_d=low_d, start_idx=start_idx
                )
                idx_gpu = sample_idx(
                    x_gpu, k, h=gpu_h, low_d=low_d, start_idx=start_idx
                )
                idx_gpu_base = sample_idx_baseline(
                    x_gpu, k, h=gpu_h, low_d=low_d, start_idx=start_idx
                )

                # Coverage metric (mean min-distance) vs CPU and GPU baseline.
                idx_cpu_gpu = idx_cpu.to(x_gpu.device)
                mean_cpu, _ = _coverage_metric(x_gpu, idx_cpu_gpu)
                mean_base, _ = _coverage_metric(x_gpu, idx_gpu_base)
                mean_opt, _ = _coverage_metric(x_gpu, idx_gpu)

                speedup_cpu = cpu_ms / gpu_ms if gpu_ms > 0 else float("inf")
                speedup_base = gpu_base_ms / gpu_ms if gpu_ms > 0 else float("inf")

                results.append(
                    {
                        "n": n,
                        "d": d,
                        "k": k,
                        "mean_cpu": mean_cpu,
                        "mean_base": mean_base,
                        "mean_gpu": mean_opt,
                        "cpu_ms": cpu_ms,
                        "gpu_base_ms": gpu_base_ms,
                        "gpu_ms": gpu_ms,
                        "speedup_cpu": speedup_cpu,
                        "speedup_base": speedup_base,
                    }
                )
                if verbose:
                    print(
                        "  -> Done. "
                        f"CPU {cpu_ms:.3f} ms | "
                        f"GPU baseline {gpu_base_ms:.3f} ms | "
                        f"GPU {gpu_ms:.3f} ms | "
                        f"MeanCPU {mean_cpu:.3e} | "
                        f"MeanBase {mean_base:.3e} | "
                        f"MeanGPU {mean_opt:.3e} | "
                        f"CPU/GPU x{speedup_cpu:.2f} | "
                        f"GPU baseline/GPU x{speedup_base:.2f}"
                    )

        print(_format_table(results))


if __name__ == "__main__":
    main()
