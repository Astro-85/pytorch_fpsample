# PyTorch FPSample

Efficient **farthest point sampling (FPS)** for PyTorch, adapted from [fpsample](https://github.com/leonardodalinky/fpsample).

This project provides bucket-based FPS on both CPU and GPU. The GPU path is optimized for high-dimensional sampling (e.g., feature embeddings).

---

## Installation

### 1) Install PyTorch (required)

Install PyTorch using the official instructions for your platform/CUDA:
- https://pytorch.org/get-started/locally/

### 2) Install `torch_fpsample`

```bash
pip install --no-build-isolation git+https://github.com/Astro-85/pytorch_fpsample

# Build locally
pip install --no-build-isolation .
````

---

## Usage

```python
import torch
import torch_fpsample

x = torch.rand(64, 2048, 256)

# Random sample
sampled_points, indices = torch_fpsample.sample(x, 1024)

# Random sample with specific tree height
sampled_points, indices = torch_fpsample.sample(x, 1024, h=3)

# Random sample with start point index (int)
sampled_points, indices = torch_fpsample.sample(x, 1024, start_idx=0)

# For high-dimensional embeddings on CUDA, set low_d for faster bucketing
sampled_points, indices = torch_fpsample.sample(x, 1024, h=8, low_d=8)

# Indices-only
indices = torch_fpsample.sample(x, 1024, return_points=False)
# (equivalently)
indices = torch_fpsample.sample_idx(x, 1024)

# Masked sampling: only sample from valid points (mask shape [B, N])
mask = torch.ones(x.shape[:-1], dtype=torch.bool)
mask[:, 1000:] = False  # e.g. padding
sampled_points, indices = torch_fpsample.sample(x, 512, mask=mask)

print(sampled_points.size(), indices.size())
# torch.Size([64, 1024, 3]) torch.Size([64, 1024])
```

---

## Performance comparison

Comparison includes CPU, a vanilla GPU FPS baseline, and our bucketed GPU implementation.

* **N**: number of input points
* **D**: point dimension
* **K**: number of sampled points
* **CPU vs GPU (bucketed)**: `CPU_ms / GPU_bucketed_ms`
* **GPU baseline vs bucketed**: `GPU_baseline_ms / GPU_bucketed_ms`

|     N |    D |    K |  CPU (ms) | GPU baseline (ms) | GPU bucketed (ms) | CPU vs GPU (bucketed) | GPU baseline vs bucketed |
| ----: | ---: | ---: | --------: | ----------------: | ----------------: | --------------------: | -----------------------: |
|  1000 |    8 |  250 |     0.271 |             0.404 |             2.671 |                 0.10x |                    0.15x |
|  1000 | 1024 |  250 |    69.697 |            94.144 |             4.867 |                14.32x |                   19.34x |
|  1000 | 4096 |  250 |   248.521 |           378.458 |            10.614 |                23.41x |                   35.65x |
|  2000 |    8 |  500 |     1.578 |             1.299 |             5.432 |                 0.29x |                    0.24x |
|  2000 | 1024 |  500 |   213.804 |           399.292 |            11.018 |                19.41x |                   36.24x |
|  2000 | 4096 |  500 |   869.318 |          1585.913 |            33.974 |                25.59x |                   46.68x |
|  5000 |    8 | 1250 |     6.151 |             7.156 |            16.970 |                 0.36x |                    0.42x |
|  5000 | 1024 | 1250 |  1075.742 |          2483.299 |            47.459 |                22.67x |                   52.33x |
|  5000 | 4096 | 1250 |  4547.318 |         10027.665 |           154.874 |                29.36x |                   64.75x |
| 10000 |    8 | 2500 |    22.135 |            26.152 |            43.379 |                 0.51x |                    0.60x |
| 10000 | 1024 | 2500 |  4503.257 |          9959.041 |           186.622 |                24.13x |                   53.36x |
| 10000 | 4096 | 2500 | 21699.598 |         40439.047 |           645.883 |                33.60x |                   62.61x |

---

## Reference

Bucket-based FPS (QuickFPS) is proposed in the following paper:

```bibtex
@article{han2023quickfps,
  title={QuickFPS: Architecture and Algorithm Co-Design for Farthest Point Sampling in Large-Scale Point Clouds},
  author={Han, Meng and Wang, Liang and Xiao, Limin and Zhang, Hao and Zhang, Chenhao and Xu, Xiangrong and Zhu, Jianfeng},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2023},
  publisher={IEEE}
}
```

Thanks to the authors for their great work.
