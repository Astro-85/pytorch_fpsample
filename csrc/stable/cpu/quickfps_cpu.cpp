// ABI-stable CPU kernels for torch_quickfps.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "cpu/bucket_fps/wrapper.h"

namespace {

inline void check_cpu_inputs(
    const torch::stable::Tensor& x,
    const torch::stable::Tensor& start_idx,
    const torch::stable::Tensor& invalid_mask) {
    STD_TORCH_CHECK(x.dim() == 3, "x must be 3D (B,N,D)");
    STD_TORCH_CHECK(x.is_cpu(), "x must be a CPU tensor");
    STD_TORCH_CHECK(start_idx.is_cpu(), "start_idx must be on CPU");
    STD_TORCH_CHECK(invalid_mask.is_cpu(), "invalid_mask must be on CPU");
    STD_TORCH_CHECK(start_idx.scalar_type() == torch::headeronly::ScalarType::Long,
                    "start_idx must be int64");
    STD_TORCH_CHECK(invalid_mask.scalar_type() == torch::headeronly::ScalarType::Byte,
                    "invalid_mask must be uint8 (Byte)");
    STD_TORCH_CHECK(start_idx.dim() == 1, "start_idx must have shape [B]");
    STD_TORCH_CHECK(invalid_mask.dim() == 2, "invalid_mask must have shape [B,N]");
    STD_TORCH_CHECK(start_idx.size(0) == x.size(0), "start_idx B mismatch");
    STD_TORCH_CHECK(invalid_mask.size(0) == x.size(0), "invalid_mask B mismatch");
    STD_TORCH_CHECK(invalid_mask.size(1) == x.size(1), "invalid_mask N mismatch");
}

inline bool batch_has_invalid(const uint8_t* inv, int64_t N) {
    for (int64_t i = 0; i < N; ++i) {
        if (inv[i]) {
            return true;
        }
    }
    return false;
}

torch::stable::Tensor sample_idx_cpu_bucket_impl(
    const torch::stable::Tensor& x,
    int64_t k,
    int64_t h,
    const torch::stable::Tensor& start_idx,
    const torch::stable::Tensor& invalid_mask,
    int64_t /*low_d*/) {
    check_cpu_inputs(x, start_idx, invalid_mask);
    STD_TORCH_CHECK(k >= 1, "k must be >= 1");
    const int64_t B = x.size(0);
    const int64_t N = x.size(1);
    const int64_t D = x.size(2);
    STD_TORCH_CHECK(k <= N, "k must be <= N");
    STD_TORCH_CHECK(h >= 1, "h must be >= 1");

    // Match original behavior: convert to float32 contiguous.
    auto x_contig = torch::stable::contiguous(x);
    auto x_f32 = torch::stable::to(x_contig, torch::headeronly::ScalarType::Float);
    x_f32 = torch::stable::contiguous(x_f32);

    auto out = torch::stable::new_empty(start_idx, {B, k}, torch::headeronly::ScalarType::Long);
    torch::stable::fill_(out, -1.0);

    const float* x_ptr = x_f32.const_data_ptr<float>();
    const int64_t* start_ptr = start_idx.const_data_ptr<int64_t>();
    const uint8_t* inv_ptr = invalid_mask.const_data_ptr<uint8_t>();
    int64_t* out_ptr = out.mutable_data_ptr<int64_t>();

    // Parallelize across batch to match the original code's behavior.
    // Each batch writes to a disjoint slice of out, so this is thread-safe.
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t b = 0; b < B; ++b) {
        const uint8_t* inv_b = inv_ptr + b * N;
        const float* x_b = x_ptr + b * N * D;
        int64_t* out_b = out_ptr + b * k;
        const int64_t s = start_ptr[b];

        if (!batch_has_invalid(inv_b, N)) {
            // Fast path.
            bucket_fps_kdline(x_b, (size_t)N, (size_t)D, (size_t)k, (size_t)s, (size_t)h, out_b);
            continue;
        }

        // Masked path: compact valid points, run bucket_fps on compacted buffer,
        // then map indices back.
        std::vector<int64_t> valid;
        valid.reserve((size_t)N);
        for (int64_t i = 0; i < N; ++i) {
            if (!inv_b[i]) {
                valid.push_back(i);
            }
        }

        const int64_t Nv = (int64_t)valid.size();
        STD_TORCH_CHECK(Nv >= k, "not enough valid points after masking");

        std::vector<float> buf;
        buf.resize((size_t)Nv * (size_t)D);
        for (int64_t j = 0; j < Nv; ++j) {
            const int64_t src = valid[j];
            const float* src_ptr = x_b + src * D;
            float* dst_ptr = buf.data() + j * D;
            std::copy(src_ptr, src_ptr + D, dst_ptr);
        }

        int64_t s_comp = -1;
        for (int64_t j = 0; j < Nv; ++j) {
            if (valid[j] == s) {
                s_comp = j;
                break;
            }
        }
        STD_TORCH_CHECK(s_comp >= 0, "start_idx must be valid when mask is used");

        std::vector<int64_t> tmp;
        tmp.resize((size_t)k);
        bucket_fps_kdline(buf.data(), (size_t)Nv, (size_t)D, (size_t)k, (size_t)s_comp, (size_t)h, tmp.data());
        for (int64_t i = 0; i < k; ++i) {
            out_b[i] = valid[(size_t)tmp[(size_t)i]];
        }
    }
    return out;
}

// Vanilla (baseline) FPS on CPU.
torch::stable::Tensor sample_idx_cpu_baseline_impl(
    const torch::stable::Tensor& x,
    int64_t k,
    const torch::stable::Tensor& start_idx,
    const torch::stable::Tensor& invalid_mask) {
    check_cpu_inputs(x, start_idx, invalid_mask);
    STD_TORCH_CHECK(k >= 1, "k must be >= 1");
    const int64_t B = x.size(0);
    const int64_t N = x.size(1);
    const int64_t D = x.size(2);
    STD_TORCH_CHECK(k <= N, "k must be <= N");

    // Use float32 distances like the CUDA baseline.
    auto x_contig = torch::stable::contiguous(x);
    auto x_f32 = torch::stable::to(x_contig, torch::headeronly::ScalarType::Float);
    x_f32 = torch::stable::contiguous(x_f32);

    auto out = torch::stable::new_empty(start_idx, {B, k}, torch::headeronly::ScalarType::Long);
    torch::stable::fill_(out, -1.0);

    const float* x_ptr = x_f32.const_data_ptr<float>();
    const int64_t* start_ptr = start_idx.const_data_ptr<int64_t>();
    const uint8_t* inv_ptr = invalid_mask.const_data_ptr<uint8_t>();
    int64_t* out_ptr = out.mutable_data_ptr<int64_t>();

    // Parallelize across batch; keep per-thread scratch buffers.
#if defined(_OPENMP)
#pragma omp parallel
    {
        std::vector<float> min_dists;
        min_dists.resize((size_t)N);

#pragma omp for schedule(static)
        for (int64_t b = 0; b < B; ++b) {
#else
    std::vector<float> min_dists;
    min_dists.resize((size_t)N);

    for (int64_t b = 0; b < B; ++b) {
#endif
        const float* x_b = x_ptr + b * N * D;
        const uint8_t* inv_b = inv_ptr + b * N;
        int64_t* out_b = out_ptr + b * k;

        // init
        for (int64_t i = 0; i < N; ++i) {
            min_dists[(size_t)i] = inv_b[i] ? -1.0f : std::numeric_limits<float>::infinity();
        }

        int64_t cur = start_ptr[b];
        out_b[0] = cur;
        if (!inv_b[cur]) {
            min_dists[(size_t)cur] = -1.0f;
        }

        // initialize distances to the first point
        const float* c0 = x_b + cur * D;
        for (int64_t i = 0; i < N; ++i) {
            if (inv_b[i] || i == cur) {
                continue;
            }
            const float* p = x_b + i * D;
            float dist = 0.0f;
            for (int64_t d = 0; d < D; ++d) {
                float diff = p[d] - c0[d];
                dist += diff * diff;
            }
            min_dists[(size_t)i] = dist;
        }

        for (int64_t it = 1; it < k; ++it) {
            // pick farthest valid point
            float best = -1.0f;
            int64_t best_idx = -1;
            for (int64_t i = 0; i < N; ++i) {
                float v = min_dists[(size_t)i];
                if (v > best) {
                    best = v;
                    best_idx = i;
                }
            }
            STD_TORCH_CHECK(best_idx >= 0, "failed to find next point (mask may be inconsistent)");
            cur = best_idx;
            out_b[it] = cur;
            min_dists[(size_t)cur] = -1.0f; // mark selected

            const float* c = x_b + cur * D;
            for (int64_t i = 0; i < N; ++i) {
                if (min_dists[(size_t)i] < 0.0f) {
                    continue; // invalid or already selected
                }
                const float* p = x_b + i * D;
                float dist = 0.0f;
                for (int64_t d = 0; d < D; ++d) {
                    float diff = p[d] - c[d];
                    dist += diff * diff;
                }
                if (dist < min_dists[(size_t)i]) {
                    min_dists[(size_t)i] = dist;
                }
            }
        }

#if defined(_OPENMP)
        }
    }
#endif

    return out;
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(torch_quickfps, CPU, m) {
    m.impl("_sample_idx_impl", TORCH_BOX(&sample_idx_cpu_bucket_impl));
    m.impl("_sample_idx_baseline_impl", TORCH_BOX(&sample_idx_cpu_baseline_impl));
}
