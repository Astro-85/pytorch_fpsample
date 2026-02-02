import os

import torch


def _run_one(device: str) -> None:
    torch.manual_seed(0)
    # Keep sizes small so the test is fast and deterministic.
    B, N, D, K = 2, 128, 3, 32
    x = torch.randn(B, N, D, device=device, dtype=torch.float32)

    # Fix the start index so CPU/GPU runs are comparable.
    start_idx = 7
    h = 6

    import torch_fpsample

    _, idx = torch_fpsample.sample(x, K, h=h, start_idx=start_idx)
    assert idx.shape == (B, K)
    # Indices should be within range.
    assert int(idx.min()) >= 0
    assert int(idx.max()) < N
    # First index should match start_idx.
    assert torch.all(idx[:, 0] == start_idx)

    # No duplicates within each batch.
    for b in range(B):
        assert idx[b].unique().numel() == K


def test_cpu_runs():
    _run_one("cpu")


def test_gpu_matches_cpu():
    if not torch.cuda.is_available() or os.environ.get("WITH_CUDA") == "0":
        return

    torch.manual_seed(0)
    B, N, D, K = 2, 256, 3, 64
    x_cpu = torch.randn(B, N, D, device="cpu", dtype=torch.float32)
    x_gpu = x_cpu.cuda()
    start_idx = 13
    h = 7

    import torch_fpsample

    _, idx_cpu = torch_fpsample.sample(x_cpu, K, h=h, start_idx=start_idx)
    _, idx_gpu = torch_fpsample.sample(x_gpu, K, h=h, start_idx=start_idx)

    # Tie-breaking differences are extremely unlikely with random data, so we
    # expect exact match.
    assert torch.equal(idx_cpu, idx_gpu.cpu())

def test_degenerate_no_duplicates_cpu():
    torch.manual_seed(0)
    B, N, D, K = 2, 64, 3, 32
    x = torch.zeros(B, N, D, device="cpu", dtype=torch.float32)
    start_idx = 0
    h = 5
    import torch_fpsample
    _, idx = torch_fpsample.sample(x, K, h=h, start_idx=start_idx)
    assert idx.shape == (B, K)
    for b in range(B):
        assert idx[b].unique().numel() == K


def test_degenerate_no_duplicates_gpu():
    if not torch.cuda.is_available() or os.environ.get("WITH_CUDA") == "0":
        return
    torch.manual_seed(0)
    B, N, D, K = 2, 64, 3, 32
    x = torch.zeros(B, N, D, device="cuda", dtype=torch.float32)
    start_idx = 0
    h = 5
    import torch_fpsample
    _, idx = torch_fpsample.sample(x, K, h=h, start_idx=start_idx)
    assert idx.shape == (B, K)
    for b in range(B):
        assert idx[b].unique().numel() == K



def test_masked_sampling_cpu():
    torch.manual_seed(0)
    B, N, D, K = 2, 128, 3, 32
    x = torch.randn(B, N, D, device="cpu", dtype=torch.float32)

    # Only even indices are valid.
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, ::2] = True

    import torch_fpsample

    _, idx = torch_fpsample.sample(x, K, h=6, mask=mask)
    assert idx.shape == (B, K)

    # First valid (deterministic) start index should be 0.
    assert torch.all(idx[:, 0] == 0)

    # All sampled indices must be valid.
    assert torch.all(mask.gather(1, idx))

    # No duplicates.
    for b in range(B):
        assert idx[b].unique().numel() == K


def test_masked_sampling_gpu_matches_cpu():
    if not torch.cuda.is_available() or os.environ.get("WITH_CUDA") == "0":
        return

    torch.manual_seed(0)
    B, N, D, K = 2, 256, 3, 64
    x_cpu = torch.randn(B, N, D, device="cpu", dtype=torch.float32)
    x_gpu = x_cpu.cuda()

    mask_cpu = torch.zeros(B, N, dtype=torch.bool)
    mask_cpu[:, ::3] = True  # sparse valid set
    mask_gpu = mask_cpu.cuda()

    import torch_fpsample

    _, idx_cpu = torch_fpsample.sample(x_cpu, K, h=7, mask=mask_cpu)
    _, idx_gpu = torch_fpsample.sample(x_gpu, K, h=7, mask=mask_gpu)

    assert torch.equal(idx_cpu, idx_gpu.cpu())


def test_mask_too_few_points_raises_cpu():
    torch.manual_seed(0)
    B, N, D, K = 2, 64, 3, 40
    x = torch.randn(B, N, D, device="cpu", dtype=torch.float32)

    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[:, :10] = True  # only 10 valid points

    import torch_fpsample

    raised = False
    try:
        torch_fpsample.sample(x, K, h=5, mask=mask)
    except RuntimeError:
        raised = True
    assert raised
