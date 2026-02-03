# PyTorch fpsample

PyTorch efficient farthest point sampling (FPS) implementation, adopted from [fpsample](https://github.com/leonardodalinky/fpsample).

**This project provides bucket-based farthest point sampling (FPS) on both CPU and CUDA GPU.**

> [!NOTE]
> Since the PyTorch capsules the native multithread implementation, this project is expected to have a much better performance than the *fpsample* implementation.

## Installation

```bash
# Install from github
pip install git+https://github.com/leonardodalinky/pytorch_fpsample

# Build locally
pip install .
```

## Usage

```python
import torch_fpsample

x = torch.rand(64, 2048, 3)
# random sample
sampled_points, indices = torch_fpsample.sample(x, 1024)
# random sample with specific tree height
sampled_points, indices = torch_fpsample.sample(x, 1024, h=5)
# random sample with start point index (int)
sampled_points, indices = torch_fpsample.sample(x, 1024, start_idx=0)

# indices-only (avoids gather; fastest path)
indices = torch_fpsample.sample(x, 1024, return_points=False)
# or equivalently:
indices = torch_fpsample.sample_idx(x, 1024)

# masked sample: only sample from valid points (mask shape [B, N])
mask = torch.ones(x.shape[:-1], dtype=torch.bool)
mask[:, 1000:] = False  # e.g. padding
sampled_points, indices = torch_fpsample.sample(x, 512, mask=mask)

> sampled_points.size(), indices.size()
Size([64, 1024, 3]), Size([64, 1024])
```

### CUDA support

This package builds a CUDA extension when a CUDA toolchain is available (i.e., `nvcc` / `CUDA_HOME`).

- Force CUDA build: `WITH_CUDA=1 pip install .` or `WITH_CUDA=1 pip install . --no-build-isolation --no-cache-dir -v`
- Force CPU-only build: `WITH_CUDA=0 pip install .`

On GPU, the sampler uses **bucket-based pruning** (spatial voxel buckets + bounding-box lower bounds) to skip most distance updates; it is **not** the standard `O(Nk)` dense FPS update.

## Reference
Bucket-based farthest point sampling (QuickFPS) is proposed in the following paper. The implementation is based on the author's Repo ([CPU](https://github.com/hanm2019/bucket-based_farthest-point-sampling_CPU) & [GPU](https://github.com/hanm2019/bucket-based_farthest-point-sampling_GPU)).
```bibtex
@article{han2023quickfps,
  title={QuickFPS: Architecture and Algorithm Co-Design for Farthest Point Sampling in Large-Scale Point Clouds},
  author={Han, Meng and Wang, Liang and Xiao, Limin and Zhang, Hao and Zhang, Chenhao and Xu, Xiangrong and Zhu, Jianfeng},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  year={2023},
  publisher={IEEE}
}
```

Thanks to the authors for their great works.

