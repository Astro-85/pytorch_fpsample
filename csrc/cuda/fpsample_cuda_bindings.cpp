#include "fpsample_cuda.h"

TORCH_LIBRARY_IMPL(torch_quickfps, CUDA, m) {
    m.impl("sample", &sample_cuda);
    m.impl("sample_idx", &sample_idx_cuda);
    m.impl("sample_baseline", &sample_cuda_baseline);
    m.impl("sample_idx_baseline", &sample_idx_cuda_baseline);
}
