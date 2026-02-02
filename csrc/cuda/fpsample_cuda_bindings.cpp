#include "fpsample_cuda.h"

TORCH_LIBRARY_IMPL(torch_fpsample, CUDA, m) { m.impl("sample", &sample_cuda); }
