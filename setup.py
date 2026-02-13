import os
import platform
import sys

from setuptools import find_packages, setup
from torch.__config__ import parallel_info
from torch.utils import cpp_extension

__version__ = "2.0.0"


def _want_cuda() -> bool:
    '''Decide whether to build the CUDA extension.

    Rules:
    - If WITH_CUDA is set, respect it ("1"/"0").
    - Otherwise, build CUDA if nvcc/CUDA_HOME is available and we're not on macOS.
    '''
    env = os.getenv("WITH_CUDA")
    if env is not None:
        return env == "1"

    # Default: enable CUDA when toolchain is available (Linux/Windows).
    if sys.platform == "darwin":
        return False
    return cpp_extension.CUDA_HOME is not None


WITH_CUDA = _want_cuda()

sources = [
    "csrc/fpsample.cpp",
    "csrc/fpsample_autograd.cpp",
    "csrc/fpsample_meta.cpp",
    "csrc/cpu/fpsample_cpu.cpp",
    "csrc/cpu/bucket_fps/wrapper.cpp",
]

if WITH_CUDA:
    sources += [
        # IMPORTANT: keep unique basenames for .cpp and .cu sources.
        # Setuptools derives the object filename from the basename; if both
        # are named fpsample_cuda.* they collide into the same .o and the
        # link step sees the object twice, causing duplicate symbols.
        "csrc/cuda/fpsample_cuda_bindings.cpp",
        "csrc/cuda/fpsample_cuda_kernels.cu",
        "csrc/cuda/fpsample_cuda_kernels_baseline.cu",
    ]

extra_compile_args = {"cxx": ["-O3"]}
extra_link_args = []

# OpenMP
info = parallel_info()
if "backend: OpenMP" in info and "OpenMP not found" not in info and sys.platform != "darwin":
    extra_compile_args["cxx"] += ["-DAT_PARALLEL_OPENMP"]
    if sys.platform == "win32":
        extra_compile_args["cxx"] += ["/openmp"]
    else:
        extra_compile_args["cxx"] += ["-fopenmp"]
else:
    print("Compiling without OpenMP...")

# CUDA flags
if WITH_CUDA:
    extra_compile_args["nvcc"] = [
        "-O3",
        "--use_fast_math",
        "-lineinfo",
    ]

# Compile for mac arm64
if sys.platform == "darwin":
    extra_compile_args["cxx"] += ["-D_LIBCPP_DISABLE_AVAILABILITY"]
    if platform.machine() == "arm64":
        extra_compile_args["cxx"] += ["-arch", "arm64"]
        extra_link_args += ["-arch", "arm64"]

if WITH_CUDA:
    ext_modules = [
        cpp_extension.CUDAExtension(
            name="torch_quickfps._core",
            include_dirs=["csrc"],
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
else:
    ext_modules = [
        cpp_extension.CppExtension(
            name="torch_quickfps._core",
            include_dirs=["csrc"],
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

setup(
    name="torch_quickfps",
    version=__version__,
    author="Leonard Lin",
    author_email="leonard.keilin@gmail.com",
    description="PyTorch bucket-based farthest point sampling (CPU + CUDA).",
    ext_modules=ext_modules,
    keywords=["pytorch", "farthest", "furthest", "sampling", "sample", "fps", "quickfps"],
    packages=find_packages(),
    package_data={"": ["*.pyi"]},
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0"],
)
