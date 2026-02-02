#include <ATen/cuda/CUDAContext.h>
// NOTE: ATen/cuda/CUDAGuard.h was removed in newer PyTorch versions.
// Use the c10 header instead (works for both host and nvcc compilation).
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "../utils.h"
#include "fpsample_cuda.h"

namespace {

#define CUDA_CHECK(err)                                                      \
    do {                                                                     \
        cudaError_t err__ = (err);                                           \
        TORCH_CHECK(err__ == cudaSuccess, "CUDA error: ", cudaGetErrorString(err__)); \
    } while (0)

__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
    int* addr_as_i = (int*)addr;
    int old = *addr_as_i;
    while (true) {
        float old_f = __int_as_float(old);
        if (old_f <= value) return old_f;
        int assumed = old;
        int new_i = __float_as_int(value);
        old = atomicCAS(addr_as_i, assumed, new_i);
        if (old == assumed) return __int_as_float(old);
    }
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    int* addr_as_i = (int*)addr;
    int old = *addr_as_i;
    while (true) {
        float old_f = __int_as_float(old);
        if (old_f >= value) return old_f;
        int assumed = old;
        int new_i = __float_as_int(value);
        old = atomicCAS(addr_as_i, assumed, new_i);
        if (old == assumed) return __int_as_float(old);
    }
}

__device__ __forceinline__ float sqr(float x) { return x * x; }

// Pack (dist, idx) into a single 64-bit key so the pair stays consistent.
//
// Why: updating `bucket_best_dist` with an atomicMax and then updating
// `bucket_best_idx` separately is racy — a thread with a smaller distance can
// overwrite the index after another thread has already written a larger max.
// Packing avoids this by updating (dist, idx) together.
//
// For non-negative floats, IEEE-754 bit patterns are monotonic with respect to
// the numeric value, so comparing the packed key as an unsigned integer is
// equivalent to comparing distances first, then breaking ties by index.
__device__ __forceinline__ unsigned long long pack_best_key(float dist, int32_t idx) {
    // Use 0 as an *invalid* sentinel key.
    // To ensure valid keys are never 0, we store (float_bits(dist) + 1) in the high 32 bits.
    // This preserves monotonic ordering for non-negative finite floats.
    unsigned int dbits = 0u;
    if (isfinite(dist) && dist >= 0.0f) {
        dbits = __float_as_uint(dist) + 1u;
    }
    unsigned int ibits = (unsigned int)idx;
    return (static_cast<unsigned long long>(dbits) << 32) | static_cast<unsigned long long>(ibits);
}

__device__ __forceinline__ float unpack_best_dist(unsigned long long key) {
    unsigned int dbits = static_cast<unsigned int>(key >> 32);
    if (dbits == 0u) return 0.0f; // invalid sentinel
    return __uint_as_float(dbits - 1u);
}

__device__ __forceinline__ int32_t unpack_best_idx(unsigned long long key) {
    return static_cast<int32_t>(key & 0xffffffffull);
}

__device__ __forceinline__ unsigned long long atomicMaxU64(unsigned long long* addr,
                                                           unsigned long long value) {
    unsigned long long old = *addr;
    while (old < value) {
        unsigned long long assumed = old;
        old = atomicCAS(addr, assumed, value);
        if (old == assumed) break;
    }
    return old;
}

__global__ void compute_bucket_id_kernel(
    const float* __restrict__ x, // [B,N,D]
    const float* __restrict__ minv, // [B,p]
    const float* __restrict__ maxv, // [B,p]
    int B, int N, int D, int p, int q,
    int32_t* __restrict__ bucket_id // [B,N]
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || i >= N) return;

    const float* xb = x + ((int64_t)b * N * D);
    // p <= 3 (by construction), and we only bucket on first p dims.
    int32_t key = 0;

    if (p == 1) {
        float mn = minv[b * p + 0];
        float mx = maxv[b * p + 0];
        float range = mx - mn;
        range = (range > 1e-12f) ? range : 1.0f;
        float v = (xb[(int64_t)i * D + 0] - mn) / range;
        v = fminf(fmaxf(v, 0.0f), 0.999999f);
        int32_t ix = (int32_t)floorf(v * q);
        ix = min(max(ix, 0), q - 1);
        key = ix;
    } else if (p == 2) {
        float mn0 = minv[b * p + 0], mx0 = maxv[b * p + 0];
        float mn1 = minv[b * p + 1], mx1 = maxv[b * p + 1];
        float r0 = mx0 - mn0; r0 = (r0 > 1e-12f) ? r0 : 1.0f;
        float r1 = mx1 - mn1; r1 = (r1 > 1e-12f) ? r1 : 1.0f;
        float v0 = (xb[(int64_t)i * D + 0] - mn0) / r0;
        float v1 = (xb[(int64_t)i * D + 1] - mn1) / r1;
        v0 = fminf(fmaxf(v0, 0.0f), 0.999999f);
        v1 = fminf(fmaxf(v1, 0.0f), 0.999999f);
        int32_t ix = (int32_t)floorf(v0 * q);
        int32_t iy = (int32_t)floorf(v1 * q);
        ix = min(max(ix, 0), q - 1);
        iy = min(max(iy, 0), q - 1);
        key = ix + q * iy;
    } else { // p == 3
        float mn0 = minv[b * p + 0], mx0 = maxv[b * p + 0];
        float mn1 = minv[b * p + 1], mx1 = maxv[b * p + 1];
        float mn2 = minv[b * p + 2], mx2 = maxv[b * p + 2];
        float r0 = mx0 - mn0; r0 = (r0 > 1e-12f) ? r0 : 1.0f;
        float r1 = mx1 - mn1; r1 = (r1 > 1e-12f) ? r1 : 1.0f;
        float r2 = mx2 - mn2; r2 = (r2 > 1e-12f) ? r2 : 1.0f;
        float v0 = (xb[(int64_t)i * D + 0] - mn0) / r0;
        float v1 = (xb[(int64_t)i * D + 1] - mn1) / r1;
        float v2 = (xb[(int64_t)i * D + 2] - mn2) / r2;
        v0 = fminf(fmaxf(v0, 0.0f), 0.999999f);
        v1 = fminf(fmaxf(v1, 0.0f), 0.999999f);
        v2 = fminf(fmaxf(v2, 0.0f), 0.999999f);
        int32_t ix = (int32_t)floorf(v0 * q);
        int32_t iy = (int32_t)floorf(v1 * q);
        int32_t iz = (int32_t)floorf(v2 * q);
        ix = min(max(ix, 0), q - 1);
        iy = min(max(iy, 0), q - 1);
        iz = min(max(iz, 0), q - 1);
        key = ix + q * (iy + q * iz);
    }

    bucket_id[b * (int64_t)N + i] = key;
}

__global__ void bbox_counts_kernel(
    const float* __restrict__ x, // [B,N,D]
    const int32_t* __restrict__ bucket_id, // [B,N]
    int B, int N, int D, int p, int key_range,
    float* __restrict__ bbox_min, // [B,key_range,p]
    float* __restrict__ bbox_max, // [B,key_range,p]
    int32_t* __restrict__ bucket_count // [B,key_range]
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || i >= N) return;

    int32_t key = bucket_id[b * (int64_t)N + i];
    if ((int)key < 0 || (int)key >= key_range) return;

    atomicAdd(&bucket_count[b * key_range + key], 1);

    const float* xb = x + ((int64_t)b * N * D);
    float* bmin = bbox_min + ((int64_t)b * key_range + key) * p;
    float* bmax = bbox_max + ((int64_t)b * key_range + key) * p;

    for (int d = 0; d < p; d++) {
        float v = xb[(int64_t)i * D + d];
        atomicMinFloat(&bmin[d], v);
        atomicMaxFloat(&bmax[d], v);
    }
}

__global__ void init_best_and_mindist_kernel(
    const float* __restrict__ x, // [B,N,D]
    const int32_t* __restrict__ bucket_id, // [B,N]
    const int64_t* __restrict__ start_idx, // [B]
    const uint8_t* __restrict__ selected_mask, // [B,N] (1=not eligible)
    int B, int N, int D, int key_range,
    float* __restrict__ min_dist, // [B,N]
    int64_t* __restrict__ bucket_best_key // [B,key_range]
) {
    int b = blockIdx.y;
    if (b >= B) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool in_range = (i < N);

    const float* xb = x + ((int64_t)b * N * D);
    int64_t sidx = start_idx[b];
    sidx = (sidx < 0) ? 0 : ((sidx >= N) ? (N - 1) : sidx);

    extern __shared__ float sh_ref[];
    // Load reference point into shared memory once per block.
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        sh_ref[d] = xb[(int64_t)sidx * D + d];
    }
    __syncthreads();

    if (!in_range) return;

    float dist = 0.0f;
    const float* xi = xb + (int64_t)i * D;
    for (int d = 0; d < D; d++) {
        dist += sqr(xi[d] - sh_ref[d]);
    }
    min_dist[b * (int64_t)N + i] = dist;

    if ((int64_t)i == sidx) return;

    if (selected_mask && selected_mask[(int64_t)b * N + i]) return;

    int32_t key = bucket_id[b * (int64_t)N + i];
    if ((int)key < 0 || (int)key >= key_range) return;

    unsigned long long* best_k = (unsigned long long*)(&bucket_best_key[b * (int64_t)key_range + key]);
    atomicMaxU64(best_k, pack_best_key(dist, (int32_t)i));
}

__global__ void reduce_bucket_best_kernel(
    const int64_t* __restrict__ bucket_best_key, // [B,key_range]
    const int32_t* __restrict__ bucket_count, // [B,key_range]
    const uint8_t* __restrict__ selected_mask, // [B,N]
    int B, int N, int key_range,
    int64_t* __restrict__ out_idx, // [B]
    float* __restrict__ out_dist // [B]
) {
    int b = blockIdx.x;
    if (b >= B) return;

    // Each block handles one batch
    unsigned long long local_best_k = 0ull;

    for (int j = threadIdx.x; j < key_range; j += blockDim.x) {
        int32_t cnt = bucket_count[b * key_range + j];
        if (cnt <= 0) continue;
        unsigned long long k = (unsigned long long)bucket_best_key[b * (int64_t)key_range + j];
        if (k == 0ull) continue;
        int32_t idx = unpack_best_idx(k);
        if (selected_mask && selected_mask[(int64_t)b * N + idx]) continue;
        if (k > local_best_k) local_best_k = k;
    }

    __shared__ unsigned long long sh_best_k[256];
    int t = threadIdx.x;
    if (t < 256) sh_best_k[t] = local_best_k;
    __syncthreads();

    // Reduce assuming blockDim.x <= 256
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            if (sh_best_k[t + stride] > sh_best_k[t]) sh_best_k[t] = sh_best_k[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) {
        unsigned long long best_k = sh_best_k[0];
        if (best_k == 0ull && selected_mask) {
            // Fallback: pick the first unselected index (degenerate cases / stale bucket bests).
            int64_t fb = 0;
            for (int64_t i = 0; i < (int64_t)N; ++i) {
                if (!selected_mask[(int64_t)b * N + i]) { fb = i; break; }
            }
            out_idx[b] = fb;
            out_dist[b] = 0.0f;
        } else {
            out_idx[b] = (int64_t)unpack_best_idx(best_k);
            out_dist[b] = unpack_best_dist(best_k);
        }
    }
}



__global__ void mark_selected_kernel(
    const int64_t* __restrict__ idx, // [B]
    int B, int N,
    uint8_t* __restrict__ selected_mask // [B,N]
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    int64_t i = idx[b];
    i = (i < 0) ? 0 : ((i >= N) ? (N - 1) : i);
    selected_mask[(int64_t)b * N + i] = 1;
}

__global__ void force_active_bucket_kernel(
    const int64_t* __restrict__ ref_idx, // [B]
    const int32_t* __restrict__ bucket_id, // [B,N]
    int B, int N, int key_range,
    uint8_t* __restrict__ active_mask // [B,key_range]
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    int64_t i = ref_idx[b];
    i = (i < 0) ? 0 : ((i >= N) ? (N - 1) : i);
    int32_t key = bucket_id[(int64_t)b * N + i];
    if ((int)key < 0 || (int)key >= key_range) return;
    active_mask[(int64_t)b * key_range + key] = 1;
}

__global__ void active_mask_kernel(
    const float* __restrict__ x, // [B,N,D]
    const int64_t* __restrict__ ref_idx, // [B]
    const float* __restrict__ bbox_min, // [B,key_range,p]
    const float* __restrict__ bbox_max, // [B,key_range,p]
    const int64_t* __restrict__ bucket_best_key, // [B,key_range]
    const int32_t* __restrict__ bucket_count, // [B,key_range]
    int B, int N, int D, int p, int key_range,
    uint8_t* __restrict__ active_mask // [B,key_range]
) {
    int b = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || j >= key_range) return;

    int32_t cnt = bucket_count[b * key_range + j];
    if (cnt <= 0) {
        active_mask[b * (int64_t)key_range + j] = 0;
        return;
    }

    float lastmax = unpack_best_dist((unsigned long long)bucket_best_key[b * (int64_t)key_range + j]);
    if (!(lastmax > 0.0f)) { // if <= 0, it can only stay <=0; no need to update
        active_mask[b * (int64_t)key_range + j] = 0;
        return;
    }

    int64_t ridx = ref_idx[b];
    ridx = (ridx < 0) ? 0 : ((ridx >= N) ? (N - 1) : ridx);

    const float* xb = x + ((int64_t)b * N * D);
    float ref0 = (p >= 1) ? xb[(int64_t)ridx * D + 0] : 0.0f;
    float ref1 = (p >= 2) ? xb[(int64_t)ridx * D + 1] : 0.0f;
    float ref2 = (p >= 3) ? xb[(int64_t)ridx * D + 2] : 0.0f;

    const float* bmin = bbox_min + ((int64_t)b * key_range + j) * p;
    const float* bmax = bbox_max + ((int64_t)b * key_range + j) * p;

    float bound = 0.0f;
    if (p >= 1) {
        float dd = 0.0f;
        if (ref0 > bmax[0]) dd = ref0 - bmax[0];
        else if (ref0 < bmin[0]) dd = bmin[0] - ref0;
        bound += dd * dd;
    }
    if (p >= 2) {
        float dd = 0.0f;
        if (ref1 > bmax[1]) dd = ref1 - bmax[1];
        else if (ref1 < bmin[1]) dd = bmin[1] - ref1;
        bound += dd * dd;
    }
    if (p >= 3) {
        float dd = 0.0f;
        if (ref2 > bmax[2]) dd = ref2 - bmax[2];
        else if (ref2 < bmin[2]) dd = bmin[2] - ref2;
        bound += dd * dd;
    }

    // Safe pruning: if the lower bound to any point in the bucket is >= current
    // bucket maximum min-distance, then distances in this bucket won't change.
    active_mask[b * (int64_t)key_range + j] = (bound < lastmax) ? 1 : 0;
}

__global__ void reset_active_bucket_best_kernel(
    const uint8_t* __restrict__ active_mask, // [B,key_range]
    const int32_t* __restrict__ bucket_count, // [B,key_range]
    int B, int key_range,
    int64_t* __restrict__ bucket_best_key // [B,key_range]
) {
    int b = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || j >= key_range) return;

    if (bucket_count[b * key_range + j] <= 0) return;
    if (active_mask[b * (int64_t)key_range + j]) {
        bucket_best_key[b * (int64_t)key_range + j] = 0;
    }
}

__global__ void update_points_kernel(
    const float* __restrict__ x, // [B,N,D]
    const int32_t* __restrict__ bucket_id, // [B,N]
    const uint8_t* __restrict__ active_mask, // [B,key_range]
    const uint8_t* __restrict__ selected_mask, // [B,N]
    const int64_t* __restrict__ ref_idx, // [B]
    int B, int N, int D, int key_range,
    float* __restrict__ min_dist, // [B,N]
    int64_t* __restrict__ bucket_best_key // [B,key_range]
) {
    int b = blockIdx.y;
    if (b >= B) return;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool in_range = (i < N);

    const float* xb = x + ((int64_t)b * N * D);

    int64_t ridx = ref_idx[b];
    ridx = (ridx < 0) ? 0 : ((ridx >= N) ? (N - 1) : ridx);

    // Load reference point into shared memory once per block.
    extern __shared__ float sh_ref[];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        sh_ref[d] = xb[(int64_t)ridx * D + d];
    }
    __syncthreads();

    if (!in_range) return;

    int32_t key = bucket_id[b * (int64_t)N + i];
    if ((int)key < 0 || (int)key >= key_range) return;

    if (!active_mask[b * (int64_t)key_range + key]) {
        // No changes needed for this bucket.
        return;
    }

    if (selected_mask && selected_mask[b * (int64_t)N + i]) {
        // Never re-select already sampled points.
        return;
    }

    float dist = 0.0f;
    const float* xi = xb + (int64_t)i * D;
    for (int d = 0; d < D; d++) {
        dist += sqr(xi[d] - sh_ref[d]);
    }

    float old = min_dist[b * (int64_t)N + i];
    float nd = (dist < old) ? dist : old;
    min_dist[b * (int64_t)N + i] = nd;

    unsigned long long* best_k = (unsigned long long*)(&bucket_best_key[b * (int64_t)key_range + key]);
    atomicMaxU64(best_k, pack_best_key(nd, (int32_t)i));
}

} // namespace

std::tuple<torch::Tensor, torch::Tensor> sample_cuda(
    const torch::Tensor &x, int64_t k, torch::optional<int64_t> h,
    torch::optional<int64_t> start_idx,
    torch::optional<torch::Tensor> mask) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor, but found on ", x.device());
    // Make sure we launch kernels on the same CUDA device as the input tensor.
    c10::cuda::CUDAGuard device_guard(x.device());
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims, but got size: ", x.sizes());
    TORCH_CHECK(k >= 1, "k must be >= 1, but got ", k);

    // (device_guard already set above)

    auto [old_size, x_reshaped_raw] = bnorm_reshape(x);
    auto x_reshaped = x_reshaped_raw.to(torch::kFloat32).contiguous(); // [B,N,D]
    const auto B = (int)x_reshaped.size(0);
    const auto N = (int)x_reshaped.size(1);
    const auto D = (int)x_reshaped.size(2);

    TORCH_CHECK(k <= N, "k must be <= N. Got k=", k, " N=", N);

    // Interpret h as a "bucket resolution" knob, similar to tree height in CPU.
    // We bucket on the first p spatial dims (p = min(D,3)) into q bins per dim,
    // with q chosen such that q^p ~= 2^h, capped to keep key_range manageable.
    int64_t height = h.value_or(5);
    height = std::max<int64_t>(1, height);

    int p = std::min<int>(D, 3);
    int q = 2;
    if (p == 1) {
        q = (int)std::min<int64_t>(4096, (1LL << std::min<int64_t>(12, height)));
    } else if (p == 2) {
        int64_t e = (height + 1) / 2;
        e = std::min<int64_t>(6, e); // cap q at 64
        q = 1 << (int)e;
    } else { // p == 3
        int64_t e = (height + 2) / 3;
        e = std::min<int64_t>(4, e); // cap q at 16
        q = 1 << (int)e;
    }
    int key_range = (p == 1) ? q : ((p == 2) ? (q * q) : (q * q * q));
    TORCH_CHECK(key_range >= 1 && key_range <= 4096,
                "Internal error: key_range out of supported range: ", key_range);

    auto opts_i32 = x_reshaped.options().dtype(torch::kInt32);
    auto opts_f32 = x_reshaped.options().dtype(torch::kFloat32);
    auto opts_u8  = x_reshaped.options().dtype(torch::kUInt8);

    // Optional validity mask: shape (*, N), where True indicates a valid point.
    // Internally we convert it into `selected_mask` where 1 means "not eligible" (already selected OR masked out).
    torch::Tensor mask_b;
    torch::Tensor selected_mask;
    if (mask.has_value() && mask.value().defined()) {
        auto m = mask.value();
        TORCH_CHECK(m.is_cuda(), "mask must be a CUDA tensor when x is CUDA");
        TORCH_CHECK(m.device() == x.device(), "mask must be on the same device as x");
        TORCH_CHECK(m.scalar_type() == torch::kBool || m.scalar_type() == torch::kUInt8,
                    "mask must have dtype bool or uint8, but got ", m.scalar_type());
        TORCH_CHECK(m.numel() == (int64_t)B * (int64_t)N,
                    "mask must have shape (*, N) matching x's batch/point dims. Expected numel=", (int64_t)B * (int64_t)N,
                    " but got numel=", m.numel());
        mask_b = m.to(torch::kBool).contiguous().view({B, N});

        // Ensure every batch has at least k valid points.
        auto counts = mask_b.sum(1); // [B]
        auto min_valid = std::get<0>(counts.min(0)).item<int64_t>();
        TORCH_CHECK(min_valid >= k,
                    "mask has fewer than k valid points in at least one batch. min_valid=", min_valid, " k=", k);

        // selected_mask == 1 means "not eligible".
        selected_mask = (~mask_b).to(torch::kUInt8).contiguous();
    } else {
        selected_mask = torch::zeros({B, N}, opts_u8);
    }

    // Start index per batch
torch::Tensor start_idx_t;
if (start_idx.has_value()) {
    int64_t s = start_idx.value();
    TORCH_CHECK(s >= 0 && s < N, "start_idx out of range: ", s, " for N=", N);
    if (mask_b.defined()) {
        auto ok = mask_b.index({at::indexing::Slice(), s}).all().item<bool>();
        TORCH_CHECK(ok, "mask disallows start_idx=", s, " in at least one batch");
    }
    start_idx_t = torch::full({B}, s, x_reshaped.options().dtype(torch::kInt64));
} else {
    if (mask_b.defined()) {
        // Deterministic: choose the first valid point per batch.
        start_idx_t = mask_b.to(torch::kInt64).argmax(1);
    } else {
        start_idx_t = torch::randint(0, N, {B}, x_reshaped.options().dtype(torch::kInt64));
    }
}

    // Compute per-batch bounding box for bucketing over the first p dims.
    // minv/maxv have shape [B, p] and are used to normalize coordinates into q bins per dim.
    Tensor x_p = x_reshaped.narrow(2, 0, p); // [B, N, p]
    Tensor minv = std::get<0>(x_p.min(1)).contiguous(); // [B, p]
    Tensor maxv = std::get<0>(x_p.max(1)).contiguous(); // [B, p]

    // Bucket metadata
    Tensor bucket_id = torch::empty({B, N}, opts_i32);
    Tensor bbox_min = torch::full({B, key_range, p},
                                  std::numeric_limits<float>::infinity(), opts_f32);
    Tensor bbox_max = torch::full({B, key_range, p},
                                  -std::numeric_limits<float>::infinity(), opts_f32);
    Tensor bucket_count = torch::zeros({B, key_range}, opts_i32);

    auto opts_i64 = x_reshaped.options().dtype(torch::kInt64);
    Tensor bucket_best_key = torch::zeros({B, key_range}, opts_i64);

    Tensor active_mask = torch::zeros({B, key_range}, opts_u8);
    Tensor min_dist = torch::empty({B, N}, opts_f32);


Tensor out_indices = torch::empty({B, (int)k}, x_reshaped.options().dtype(torch::kInt64));
    // First sample is the start point.
    out_indices.select(1, 0).copy_(start_idx_t);

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    // Track selected points to guarantee unique indices even in degenerate cases.
    dim3 gridB((B + 255) / 256);
    mark_selected_kernel<<<gridB, 256, 0, stream>>>(
        start_idx_t.data_ptr<int64_t>(), B, N, selected_mask.data_ptr<uint8_t>());
    CUDA_CHECK(cudaGetLastError());

    // 1) Bucket assignment
    dim3 block(256);
    dim3 grid_bucket((N + block.x - 1) / block.x, B);
    compute_bucket_id_kernel<<<grid_bucket, block, 0, stream>>>(
        x_reshaped.data_ptr<float>(),
        minv.data_ptr<float>(),
        maxv.data_ptr<float>(),
        B, N, D, p, q,
        bucket_id.data_ptr<int32_t>());
    CUDA_CHECK(cudaGetLastError());

    // 2) Bucket bounding boxes + counts
    bbox_counts_kernel<<<grid_bucket, block, 0, stream>>>(
        x_reshaped.data_ptr<float>(),
        bucket_id.data_ptr<int32_t>(),
        B, N, D, p, key_range,
        bbox_min.data_ptr<float>(),
        bbox_max.data_ptr<float>(),
        bucket_count.data_ptr<int32_t>());
    CUDA_CHECK(cudaGetLastError());

    // 3) Initialize min_dist to distance from start point, and compute per-bucket maxima.
    size_t shmem = (size_t)D * sizeof(float);
    init_best_and_mindist_kernel<<<grid_bucket, block, shmem, stream>>>(
        x_reshaped.data_ptr<float>(),
        bucket_id.data_ptr<int32_t>(),
        start_idx_t.data_ptr<int64_t>(),
        selected_mask.data_ptr<uint8_t>(),
        B, N, D, key_range,
        min_dist.data_ptr<float>(),
        bucket_best_key.data_ptr<int64_t>());
    CUDA_CHECK(cudaGetLastError());

    // Temporary buffers for reduction results
    torch::Tensor next_idx = torch::empty({B}, x_reshaped.options().dtype(torch::kInt64));
    torch::Tensor next_dist = torch::empty({B}, opts_f32);

    // Sampling loop
    for (int64_t s = 1; s < k; s++) {
        // Pick next point (argmax over buckets)
        reduce_bucket_best_kernel<<<B, 256, 0, stream>>>(
            bucket_best_key.data_ptr<int64_t>(),
            bucket_count.data_ptr<int32_t>(),
            selected_mask.data_ptr<uint8_t>(),
            B, N, key_range,
            next_idx.data_ptr<int64_t>(),
            next_dist.data_ptr<float>());
        CUDA_CHECK(cudaGetLastError());

        out_indices.select(1, (int)s).copy_(next_idx);

        mark_selected_kernel<<<gridB, 256, 0, stream>>>(
            next_idx.data_ptr<int64_t>(), B, N, selected_mask.data_ptr<uint8_t>());
        CUDA_CHECK(cudaGetLastError());

        // No need to update distances after the last selection.
        if (s == k - 1) break;

        // Compute active buckets for this ref point
        dim3 grid_bucket2((key_range + block.x - 1) / block.x, B);
        active_mask_kernel<<<grid_bucket2, block, 0, stream>>>(
            x_reshaped.data_ptr<float>(),
            next_idx.data_ptr<int64_t>(),
            bbox_min.data_ptr<float>(),
            bbox_max.data_ptr<float>(),
            bucket_best_key.data_ptr<int64_t>(),
            bucket_count.data_ptr<int32_t>(),
            B, N, D, p, key_range,
            active_mask.data_ptr<uint8_t>());
        CUDA_CHECK(cudaGetLastError());

        force_active_bucket_kernel<<<gridB, 256, 0, stream>>>(
            next_idx.data_ptr<int64_t>(),
            bucket_id.data_ptr<int32_t>(),
            B, N, key_range,
            active_mask.data_ptr<uint8_t>());
        CUDA_CHECK(cudaGetLastError());

        // Reset per-bucket maxima for active buckets so we can recompute them.
        reset_active_bucket_best_kernel<<<grid_bucket2, block, 0, stream>>>(
            active_mask.data_ptr<uint8_t>(),
            bucket_count.data_ptr<int32_t>(),
            B, key_range,
            bucket_best_key.data_ptr<int64_t>());
        CUDA_CHECK(cudaGetLastError());

        // Update min_dist for points in active buckets and recompute active bucket maxima.
        update_points_kernel<<<grid_bucket, block, shmem, stream>>>(
            x_reshaped.data_ptr<float>(),
            bucket_id.data_ptr<int32_t>(),
            active_mask.data_ptr<uint8_t>(),
            selected_mask.data_ptr<uint8_t>(),
            next_idx.data_ptr<int64_t>(),
            B, N, D, key_range,
            min_dist.data_ptr<float>(),
            bucket_best_key.data_ptr<int64_t>());
        CUDA_CHECK(cudaGetLastError());
    }

    // Gather sampled points from the original (possibly non-float32) view.
    auto gathered = torch::gather(
        x_reshaped_raw, 1,
        out_indices.view({B, (int)k, 1}).repeat({1, 1, D}));

    // Reshape back to the original batch shape.
    auto ret_tensor_sizes = old_size.vec();
    ret_tensor_sizes[ret_tensor_sizes.size() - 2] = k;
    auto ret_indices_sizes = old_size.vec();
    ret_indices_sizes.pop_back();
    ret_indices_sizes[ret_indices_sizes.size() - 1] = k;

    return std::make_tuple(
        gathered.view(ret_tensor_sizes),
        out_indices.view(ret_indices_sizes).to(torch::kLong));
}
