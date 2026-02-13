/**
 * Baseline vanilla FPS (CUDA) adapted from PyTorch3D.
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

#include "../utils.h"

namespace {

#define CUDA_CHECK(err)                                                      \
    do {                                                                     \
        cudaError_t err__ = (err);                                           \
        TORCH_CHECK(err__ == cudaSuccess, "CUDA error: ", cudaGetErrorString(err__)); \
    } while (0)

template <unsigned int block_size>
__global__ void FarthestPointSamplingKernelBaseline(
    const float* __restrict__ points, // [B, N, D]
    const int64_t* __restrict__ K, // [B]
    const int64_t* __restrict__ start_idxs, // [B]
    const uint8_t* __restrict__ mask, // [B, N] (1=valid) or nullptr
    int64_t N,
    int64_t D,
    int64_t max_k,
    int64_t* __restrict__ idxs, // [B, max_k]
    float* __restrict__ min_point_dist // [B, N]
) {
    typedef cub::BlockReduce<
        cub::KeyValuePair<int64_t, float>,
        block_size,
        cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>
        BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ int64_t selected_store;

    const int64_t b = blockIdx.x;
    const size_t tid = threadIdx.x;

    const int64_t k_n = min(K[b], max_k);

    int64_t selected = start_idxs[b];
    if (tid == 0) {
        idxs[b * max_k + 0] = selected;
    }

    for (int64_t k = 1; k < k_n; ++k) {
        int64_t max_dist_idx = 0;
        float max_dist = -1.0f;

        for (int64_t p = tid; p < N; p += block_size) {
            if (mask && mask[b * N + p] == 0) {
                continue;
            }

            float dist2 = 0.0f;
            const float* sel = points + ((b * N + selected) * D);
            const float* cur = points + ((b * N + p) * D);
            for (int64_t d = 0; d < D; ++d) {
                float diff = sel[d] - cur[d];
                dist2 += diff * diff;
            }

            float* p_min_ptr = min_point_dist + (b * N + p);
            float p_min_dist = dist2 < *p_min_ptr ? dist2 : *p_min_ptr;
            *p_min_ptr = p_min_dist;

            if (p_min_dist > max_dist) {
                max_dist = p_min_dist;
                max_dist_idx = p;
            }
        }

        selected =
            BlockReduce(temp_storage)
                .Reduce(
                    cub::KeyValuePair<int64_t, float>(max_dist_idx, max_dist),
                    cub::ArgMax(),
                    block_size)
                .key;

        if (tid == 0) {
            idxs[b * max_k + k] = selected;
            selected_store = selected;
        }
        __syncthreads();
        selected = selected_store;
    }
}

static torch::Tensor sample_cuda_indices_baseline_impl(
    const torch::Tensor& x,
    int64_t k,
    torch::optional<int64_t> start_idx,
    torch::optional<torch::Tensor> mask) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor, but found on ", x.device());
    c10::cuda::CUDAGuard device_guard(x.device());
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims, but got size: ", x.sizes());
    TORCH_CHECK(k >= 1, "k must be >= 1, but got ", k);

    auto [old_size, x_reshaped_raw] = bnorm_reshape(x);
    auto x_reshaped = x_reshaped_raw.contiguous();

    TORCH_CHECK(x_reshaped.scalar_type() == torch::kFloat32 ||
                x_reshaped.scalar_type() == torch::kFloat16 ||
                x_reshaped.scalar_type() == torch::kBFloat16,
                "x must have dtype float32/float16/bfloat16 on CUDA, but got ", x_reshaped.scalar_type());

    const int64_t B = x_reshaped.size(0);
    const int64_t N = x_reshaped.size(1);
    const int64_t D = x_reshaped.size(2);

    TORCH_CHECK(k <= N, "k must be <= N. Got k=", k, " N=", N);

    auto opts_i64 = x_reshaped.options().dtype(torch::kInt64);
    auto opts_f32 = x_reshaped.options().dtype(torch::kFloat32);
    auto opts_u8 = x_reshaped.options().dtype(torch::kUInt8);

    torch::Tensor mask_b;
    torch::Tensor mask_u8;
    if (mask.has_value() && mask.value().defined()) {
        auto m = mask.value();
        TORCH_CHECK(m.is_cuda(), "mask must be a CUDA tensor when x is CUDA");
        TORCH_CHECK(m.device() == x.device(), "mask must be on the same device as x");
        TORCH_CHECK(m.scalar_type() == torch::kBool || m.scalar_type() == torch::kUInt8,
                    "mask must have dtype bool or uint8, but got ", m.scalar_type());
        TORCH_CHECK(m.numel() == B * N,
                    "mask must have shape (*, N) matching x's batch/point dims. Expected numel=",
                    B * N, " but got numel=", m.numel());
        mask_b = m.to(torch::kBool).contiguous().view({B, N});

        auto counts = mask_b.sum(1);
        auto min_valid = std::get<0>(counts.min(0)).item<int64_t>();
        TORCH_CHECK(min_valid >= k,
                    "mask has fewer than k valid points in at least one batch. min_valid=",
                    min_valid, " k=", k);

        mask_u8 = mask_b.to(torch::kUInt8).contiguous(); // 1=valid, 0=invalid
    }

    torch::Tensor start_idx_t;
    if (start_idx.has_value()) {
        int64_t s = start_idx.value();
        TORCH_CHECK(s >= 0 && s < N, "start_idx out of range: ", s, " for N=", N);
        if (mask_b.defined()) {
            auto ok = mask_b.select(1, s).all().item<bool>();
            TORCH_CHECK(ok, "mask disallows start_idx=", s, " in at least one batch");
        }
        start_idx_t = torch::full({B}, s, opts_i64);
    } else {
        if (mask_b.defined()) {
            start_idx_t = mask_b.to(torch::kInt64).argmax(1);
        } else {
            start_idx_t = torch::randint(0, N, {B}, opts_i64);
        }
    }

    torch::Tensor x_f32 = x_reshaped.to(torch::kFloat32).contiguous();
    torch::Tensor K_t = torch::full({B}, k, opts_i64);

    auto idxs = torch::full({B, k}, -1, opts_i64);
    auto min_point_dist = torch::full({B, N}, 1e10, opts_f32);

    if (B == 0 || N == 0) {
        CUDA_CHECK(cudaGetLastError());
        auto ret_indices_sizes = old_size.vec();
        ret_indices_sizes.pop_back();
        ret_indices_sizes[ret_indices_sizes.size() - 1] = k;
        return idxs.view(ret_indices_sizes).to(torch::kLong);
    }

    const size_t blocks = (size_t)B;

    const int points_pow_2 = (N > 1) ? (int)std::log2((double)N) : 0;
    const int MAX_THREADS_PER_BLOCK = 1024;
    const size_t threads = std::max((size_t)2, std::min((size_t)(1 << points_pow_2), (size_t)MAX_THREADS_PER_BLOCK));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const float* points_ptr = x_f32.data_ptr<float>();
    const int64_t* K_ptr = K_t.data_ptr<int64_t>();
    const int64_t* start_ptr = start_idx_t.data_ptr<int64_t>();
    const uint8_t* mask_ptr = mask_u8.defined() ? mask_u8.data_ptr<uint8_t>() : nullptr;

    const int64_t max_k = k;

    switch (threads) {
        case 1024:
            FarthestPointSamplingKernelBaseline<1024>
                <<<blocks, threads, 0, stream>>>(
                    points_ptr, K_ptr, start_ptr, mask_ptr, N, D, max_k,
                    idxs.data_ptr<int64_t>(), min_point_dist.data_ptr<float>());
            break;
        case 512:
            FarthestPointSamplingKernelBaseline<512><<<blocks, threads, 0, stream>>>(
                points_ptr, K_ptr, start_ptr, mask_ptr, N, D, max_k,
                idxs.data_ptr<int64_t>(), min_point_dist.data_ptr<float>());
            break;
        case 256:
            FarthestPointSamplingKernelBaseline<256><<<blocks, threads, 0, stream>>>(
                points_ptr, K_ptr, start_ptr, mask_ptr, N, D, max_k,
                idxs.data_ptr<int64_t>(), min_point_dist.data_ptr<float>());
            break;
        case 128:
            FarthestPointSamplingKernelBaseline<128><<<blocks, threads, 0, stream>>>(
                points_ptr, K_ptr, start_ptr, mask_ptr, N, D, max_k,
                idxs.data_ptr<int64_t>(), min_point_dist.data_ptr<float>());
            break;
        case 64:
            FarthestPointSamplingKernelBaseline<64><<<blocks, threads, 0, stream>>>(
                points_ptr, K_ptr, start_ptr, mask_ptr, N, D, max_k,
                idxs.data_ptr<int64_t>(), min_point_dist.data_ptr<float>());
            break;
        case 32:
            FarthestPointSamplingKernelBaseline<32><<<blocks, threads, 0, stream>>>(
                points_ptr, K_ptr, start_ptr, mask_ptr, N, D, max_k,
                idxs.data_ptr<int64_t>(), min_point_dist.data_ptr<float>());
            break;
        case 16:
            FarthestPointSamplingKernelBaseline<16><<<blocks, threads, 0, stream>>>(
                points_ptr, K_ptr, start_ptr, mask_ptr, N, D, max_k,
                idxs.data_ptr<int64_t>(), min_point_dist.data_ptr<float>());
            break;
        case 8:
            FarthestPointSamplingKernelBaseline<8><<<blocks, threads, 0, stream>>>(
                points_ptr, K_ptr, start_ptr, mask_ptr, N, D, max_k,
                idxs.data_ptr<int64_t>(), min_point_dist.data_ptr<float>());
            break;
        case 4:
            FarthestPointSamplingKernelBaseline<4><<<blocks, threads, 0, stream>>>(
                points_ptr, K_ptr, start_ptr, mask_ptr, N, D, max_k,
                idxs.data_ptr<int64_t>(), min_point_dist.data_ptr<float>());
            break;
        case 2:
            FarthestPointSamplingKernelBaseline<2><<<blocks, threads, 0, stream>>>(
                points_ptr, K_ptr, start_ptr, mask_ptr, N, D, max_k,
                idxs.data_ptr<int64_t>(), min_point_dist.data_ptr<float>());
            break;
        default:
            FarthestPointSamplingKernelBaseline<1024>
                <<<blocks, threads, 0, stream>>>(
                    points_ptr, K_ptr, start_ptr, mask_ptr, N, D, max_k,
                    idxs.data_ptr<int64_t>(), min_point_dist.data_ptr<float>());
            break;
    }

    CUDA_CHECK(cudaGetLastError());

    auto ret_indices_sizes = old_size.vec();
    ret_indices_sizes.pop_back();
    ret_indices_sizes[ret_indices_sizes.size() - 1] = k;

    return idxs.view(ret_indices_sizes).to(torch::kLong);
}

} // namespace

// Baseline vanilla FPS entry points (not bound to a dispatcher by default).
// These mirror torch_quickfps::sample_idx and torch_quickfps::sample signatures.

torch::Tensor sample_idx_cuda_baseline(
    const torch::Tensor& x,
    int64_t k,
    torch::optional<int64_t> h,
    torch::optional<int64_t> start_idx,
    torch::optional<torch::Tensor> mask,
    torch::optional<int64_t> low_d) {

    (void)h;
    (void)low_d;
    return sample_cuda_indices_baseline_impl(x, k, start_idx, mask);
}

std::tuple<torch::Tensor, torch::Tensor> sample_cuda_baseline(
    const torch::Tensor& x,
    int64_t k,
    torch::optional<int64_t> h,
    torch::optional<int64_t> start_idx,
    torch::optional<torch::Tensor> mask,
    torch::optional<int64_t> low_d) {

    (void)h;
    (void)low_d;
    auto idx = sample_cuda_indices_baseline_impl(x, k, start_idx, mask);

    auto [old_size, x_reshaped_raw] = bnorm_reshape(x);
    auto x_reshaped = x_reshaped_raw.contiguous();
    const int64_t B = x_reshaped.size(0);
    const int64_t D = x_reshaped.size(2);

    auto idx_b = idx.view({B, (int)k});
    auto gathered = torch::gather(
        x_reshaped, 1, idx_b.unsqueeze(-1).repeat({1, 1, D}));

    auto ret_tensor_sizes = old_size.vec();
    ret_tensor_sizes[ret_tensor_sizes.size() - 2] = k;

    return std::make_tuple(gathered.view(ret_tensor_sizes), idx);
}
