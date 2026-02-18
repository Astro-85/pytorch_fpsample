import os
import platform
import sys

from setuptools import find_packages, setup
from torch.__config__ import parallel_info
from torch.utils import cpp_extension

__version__ = "2.1.0"


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

# ABI-stable build: compile only the Stable-ABI extension sources.
sources = [
    "csrc/stable/quickfps_ops.cpp",
    "csrc/stable/stable_tensor_inl.cpp",
    "csrc/stable/cpu/quickfps_cpu.cpp",
    "csrc/cpu/bucket_fps/wrapper.cpp",
]

if WITH_CUDA:
    sources += [
        "csrc/stable/cuda/quickfps_cuda.cu",
    ]

extra_compile_args = {
    "cxx": [
        "-O3",        # LibTorch Stable ABI (minimum PyTorch 2.10.0)
        "-DTORCH_TARGET_VERSION=0x020a000000000000",
    ]
}
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
        # libtorch stable ABI flag for nvcc compilation units too
        "-DTORCH_TARGET_VERSION=0x020a000000000000",
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
            py_limited_api=True,
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
            py_limited_api=True,
        )
    ]

setup(
    name="torch_quickfps",
    version=__version__,
    author="Andrew Lu",
    author_email="alu1@seas.upenn.edu",
    description="PyTorch bucket-based farthest point sampling (CPU + CUDA).",
    ext_modules=ext_modules,
    keywords=["pytorch", "farthest", "furthest", "sampling", "sample", "fps", "quickfps"],
    packages=find_packages(),
    package_data={"": ["*.pyi"]},
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    python_requires=">=3.10",
    install_requires=["torch>=2.10"],
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
