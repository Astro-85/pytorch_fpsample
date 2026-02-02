#include <torch/library.h>

#include <vector>
#include <cstdint>
#include <cstring>

#include "../utils.h"
#include "bucket_fps/wrapper.h"

using torch::Tensor;

///////////////
//           //
//    CPU    //
//           //
///////////////

static inline void check_mask_shape_cpu(const Tensor& mask, int64_t B, int64_t N) {
    TORCH_CHECK(mask.device().is_cpu(), "mask must be a CPU tensor, but found on ", mask.device());
    TORCH_CHECK(mask.scalar_type() == torch::kBool || mask.scalar_type() == torch::kUInt8,
                "mask must have dtype bool or uint8, but got ", mask.scalar_type());
    TORCH_CHECK(mask.numel() == B * N,
                "mask must have shape (*, N) matching x's batch/point dims. "
                "Expected numel=", (B * N), " but got numel=", mask.numel());
}

std::tuple<Tensor, Tensor> sample_cpu(
    const Tensor &x,
    int64_t k,
    torch::optional<int64_t> h,
    torch::optional<int64_t> start_idx,
    torch::optional<Tensor> mask_opt) {

    TORCH_CHECK(x.device().is_cpu(), "x must be a CPU tensor, but found on ", x.device());
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims, but got size: ", x.sizes());
    TORCH_CHECK(k >= 1, "k must be greater than or equal to 1, but got ", k);

    auto [old_size, x_reshaped_raw] = bnorm_reshape(x);
    auto x_reshaped = x_reshaped_raw.to(torch::kFloat32).contiguous(); // [B,N,D]

    const int64_t B = x_reshaped.size(0);
    const int64_t N = x_reshaped.size(1);
    const int64_t D = x_reshaped.size(2);

    TORCH_CHECK(k <= N, "k must be <= N. Got k=", k, " N=", N);

    auto height = h.value_or(5);
    height = std::max<int64_t>(1, height);

    // Prepare output
    Tensor ret_indices = torch::empty({B, k}, x_reshaped.options().dtype(torch::kInt64));

    // Fast path: no mask -> sample on full set, identical to original implementation.
    if (!mask_opt.has_value() || !mask_opt.value().defined()) {
        torch::Tensor cur_start_idx;
        if (start_idx.has_value()) {
            cur_start_idx = torch::full({B}, (int64_t)start_idx.value(),
                                        x_reshaped.options().dtype(torch::kInt64));
        } else {
            cur_start_idx = torch::randint(0, N, {B}, x_reshaped.options().dtype(torch::kInt64));
        }

        TORCH_CHECK(k <= N, "k must be <= N. Got k=", k, " N=", N);

        at::parallel_for(0, B, 0, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; i++) {
                bucket_fps_kdline(
                    x_reshaped[i].data_ptr<float>(), (int)N, (int)D, (int)k,
                    (int)cur_start_idx[i].item<int64_t>(), (int)height,
                    ret_indices[i].data_ptr<int64_t>());
            }
        });

        auto gathered = torch::gather(
            x_reshaped_raw, 1, ret_indices.view({B, k, 1}).repeat({1, 1, D}));

        auto ret_tensor_sizes = old_size.vec();
        ret_tensor_sizes[ret_tensor_sizes.size() - 2] = k;
        auto ret_indices_sizes = old_size.vec();
        ret_indices_sizes.pop_back();
        ret_indices_sizes[ret_indices_sizes.size() - 1] = k;

        return std::make_tuple(
            gathered.view(ret_tensor_sizes),
            ret_indices.view(ret_indices_sizes).to(torch::kLong));
    }

    // Masked path: compact valid points per batch, run bucket FPS on the compacted set,
    // and map indices back to the original [0, N) indexing.
    Tensor mask = mask_opt.value();
    check_mask_shape_cpu(mask, B, N);

    // Reshape to [B, N] using the same batch flattening as bnorm_reshape(x).
    Tensor mask_b = mask.to(torch::kBool).contiguous().view({B, N});

    // If a global start_idx is provided, it must be valid for every batch.
    if (start_idx.has_value()) {
        int64_t s = start_idx.value();
        TORCH_CHECK(s >= 0 && s < N, "start_idx out of range: ", s, " for N=", N);
        // Check validity across batches (sync-free on CPU).
        auto acc = mask_b.accessor<bool, 2>();
        for (int64_t b = 0; b < B; ++b) {
            TORCH_CHECK(acc[b][s], "mask disallows start_idx=", s, " in batch ", b);
        }
    }

    at::parallel_for(0, B, 0, [&](int64_t begin, int64_t end) {
        for (int64_t b = begin; b < end; b++) {
            const bool* mptr = mask_b[b].data_ptr<bool>();
            std::vector<int64_t> valid;
            valid.reserve((size_t)N);

            for (int64_t i = 0; i < N; ++i) {
                if (mptr[i]) valid.push_back(i);
            }

            TORCH_CHECK((int64_t)valid.size() >= k,
                        "mask has only ", valid.size(), " valid points in batch ", b,
                        " but k=", k);

            int64_t sidx_orig = start_idx.has_value() ? start_idx.value() : valid[0];

            // Find sidx in the compacted list.
            int64_t sidx_compact = -1;
            for (int64_t t = 0; t < (int64_t)valid.size(); ++t) {
                if (valid[t] == sidx_orig) { sidx_compact = t; break; }
            }
            TORCH_CHECK(sidx_compact >= 0, "internal error: start_idx not found in valid list");

            // Build compact point buffer [Nv, D]
            const float* xb = x_reshaped[b].data_ptr<float>();
            const int64_t Nv = (int64_t)valid.size();
            std::vector<float> buf((size_t)Nv * (size_t)D);

            for (int64_t t = 0; t < Nv; ++t) {
                int64_t orig = valid[t];
                std::memcpy(buf.data() + (size_t)t * (size_t)D,
                            xb + (size_t)orig * (size_t)D,
                            (size_t)D * sizeof(float));
            }

            std::vector<int64_t> out_compact((size_t)k);
            bucket_fps_kdline(
                buf.data(), (int)Nv, (int)D, (int)k,
                (int)sidx_compact, (int)height,
                out_compact.data());

            // Map back to original indices.
            int64_t* out_ptr = ret_indices[b].data_ptr<int64_t>();
            for (int64_t t = 0; t < k; ++t) {
                int64_t ci = out_compact[(size_t)t];
                TORCH_CHECK(ci >= 0 && ci < Nv, "internal error: compact index out of range");
                out_ptr[t] = valid[(size_t)ci];
            }
        }
    });

    auto gathered = torch::gather(
        x_reshaped_raw, 1, ret_indices.view({B, k, 1}).repeat({1, 1, D}));

    auto ret_tensor_sizes = old_size.vec();
    ret_tensor_sizes[ret_tensor_sizes.size() - 2] = k;
    auto ret_indices_sizes = old_size.vec();
    ret_indices_sizes.pop_back();
    ret_indices_sizes[ret_indices_sizes.size() - 1] = k;

    return std::make_tuple(
        gathered.view(ret_tensor_sizes),
        ret_indices.view(ret_indices_sizes).to(torch::kLong));
}

TORCH_LIBRARY_IMPL(torch_fpsample, CPU, m) { m.impl("sample", &sample_cpu); }
