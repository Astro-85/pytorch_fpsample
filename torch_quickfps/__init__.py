import importlib.machinery
import os.path as osp

import torch

from .fps import sample, sample_idx, sample_baseline, sample_idx_baseline

# Load the compiled library first. This registers the internal (ABI-stable)
# kernels into the dispatcher.

torch.ops.load_library(
    importlib.machinery.PathFinder().find_spec(f"_core", [osp.dirname(__file__)]).origin
)

# Register the public operators (schemas + CompositeExplicitAutograd
# implementations) in Python. These forward to the internal compiled kernels.
from . import _stable_register as _stable_register  # noqa: E402,F401

__all__ = ["sample", "sample_idx", "sample_baseline", "sample_idx_baseline"]


