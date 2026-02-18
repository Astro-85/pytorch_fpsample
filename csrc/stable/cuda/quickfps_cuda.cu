// ABI-stable CUDA implementations for torch_quickfps.
//
// This translation unit:
//  - Reuses the existing highly-optimized CUDA bucket-based FPS kernels.
//  - Exposes ABI-stable dispatcher entrypoints using the LibTorch Stable ABI.
//
// NOTE: This file is compiled with:
//   -DTORCH_TARGET_VERSION=0x020a000000000000  (PyTorch 2.10+ Stable ABI)
//   -DPy_LIMITED_API=0x03090000                (CPython abi3 wheels)

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/c/shim.h>

#include <torch/headeronly/core/DeviceType.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>

// -----------------------------------------------------------------------------
// CUDA stream shim
//
// Some PyTorch builds do not provide the declaration of
// `aoti_torch_get_current_cuda_stream` to NVCC via the stable shim header.
// The symbol is part of the AOTI C shim interface; forward-declare it here
// for robust compilation.
//
// Signature matches the AOTI C shim (device_index is int32_t).
extern "C" int32_t aoti_torch_get_current_cuda_stream(int32_t device_index, void** ret_stream);

namespace {

constexpr int kMaxBucketingP = 64;

#define CUDA_CHECK(err)                                                      \
    do {                                                                     \
        cudaError_t err__ = (err);                                           \
        STD_TORCH_CHECK(err__ == cudaSuccess, "CUDA error: ", cudaGetErrorString(err__)); \
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

// Pack (dist, idx) into a single 64-bit key so the pair stays consistent.
//
// We store (float_bits(dist) + 1) in the high 32 bits so that 0 is reserved as an invalid sentinel.
// For non-negative finite floats, IEEE-754 bit patterns are monotonic w.r.t. numeric value.
__device__ __forceinline__ unsigned long long pack_best_key(float dist, int32_t idx) {
    unsigned int dbits = 0u;
    if (isfinite(dist) && dist >= 0.0f) {
        dbits = __float_as_uint(dist) + 1u;
    }
    unsigned int ibits = (unsigned int)idx;
    return (static_cast<unsigned long long>(dbits) << 32) | static_cast<unsigned long long>(ibits);
}

__device__ __forceinline__ float unpack_best_dist(unsigned long long key) {
    unsigned int dbits = static_cast<unsigned int>(key >> 32);
    if (dbits == 0u) return 0.0f;
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

__device__ __forceinline__ unsigned long long block_reduce_best_key(
    const int64_t* __restrict__ bucket_best_key,
    const int32_t* __restrict__ bucket_count,
    uint8_t* __restrict__ selected_mask,
    int B, int N, int key_range,
    int b,
    unsigned long long* __restrict__ sh_best_k
) {
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

    int t = threadIdx.x;
    if (t < 256) sh_best_k[t] = local_best_k;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            if (sh_best_k[t + stride] > sh_best_k[t]) sh_best_k[t] = sh_best_k[t + stride];
        }
        __syncthreads();
    }

    return sh_best_k[0];
}

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t v) {
    return static_cast<float>(v);
}

// -------------------------
// Bucketing kernels
// -------------------------

// Cheap, geometry-agnostic projection for high-D embeddings.
//
// We use p Rademacher sign vectors and scale them so the projection is
// non-expansive (1-Lipschitz) in L2:
//   y = x @ R, with ||R||_2 <= ||R||_F = 1.
//
// Concretely, each projection vector has entries in {+1,-1} and is scaled by
// 1/sqrt(D*p). This guarantees ||y(x)-y(z)|| <= ||x-z||, which keeps the
// pruning bound in active mask pruning safe.
__device__ __forceinline__ uint32_t mixbits(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ float rademacher_sign(uint32_t seed, int proj, int dim) {
    uint32_t x = (uint32_t)dim;
    x ^= seed + 0x9e3779b9u + (uint32_t)proj * 0x85ebca6bu;
    x = mixbits(x);
    return (x & 1u) ? 1.0f : -1.0f;
}

template <typename scalar_t>
__global__ void project_rademacher_kernel(
    const scalar_t* __restrict__ x, // [B,N,D]
    int B, int N, int D, int p,
    uint32_t seed,
    float* __restrict__ coords // [B,N,p] float32
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || i >= N) return;

    const scalar_t* xb = x + ((int64_t)b * N * D);
    const scalar_t* xi = xb + (int64_t)i * D;
    // NOTE: We keep p small (user-controlled low_d) so this remains cheap.
    constexpr int MAX_P = kMaxBucketingP;
    // Scale so ||R||_F = 1 => ||R||_2 <= 1, hence projection is non-expansive.
    float inv_norm = rsqrtf((float)D * (float)p);

    float acc[MAX_P];
    #pragma unroll
    for (int j = 0; j < MAX_P; ++j) acc[j] = 0.0f;

    for (int d = 0; d < D; ++d) {
        float v = to_float(xi[d]);
        // Unroll the inner loop when p is small; p is runtime, so the compiler
        // won't fully unroll, but MAX_P is small enough to keep overhead low.
        for (int j = 0; j < p; ++j) {
            acc[j] += rademacher_sign(seed, j, d) * v;
        }
    }

    float* out = coords + ((int64_t)b * N + i) * p;
    for (int j = 0; j < p; ++j) {
        out[j] = acc[j] * inv_norm;
    }
}

// -------------------------
// GPU-side adaptive KD-tree builder (level-by-level)
// -------------------------
//
// Build a complete binary KD-tree of fixed `height` in the bucketing space
// (dimension p). Instead of sorting, we iterate level-by-level:
//   1) each point carries a node id in [0, 2^depth)
//   2) we compute per-node bbox + sums in parallel (excluding invalid points)
//   3) choose split dim = widest bbox extent; split threshold = mean along that dim
//   4) update each point's node id by comparing to the threshold
//
// This avoids GPU->CPU syncs/copies and keeps the FPS update path fully parallel.

__global__ void kdtree_reset_stats_kernel(
    int B,
    int max_nodes,
    int nodes,
    int p,
    float* __restrict__ node_min, // [B,max_nodes,p]
    float* __restrict__ node_max, // [B,max_nodes,p]
    float* __restrict__ node_sum, // [B,max_nodes,p]
    int32_t* __restrict__ node_count // [B,max_nodes]
) {
    int b = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    int total = nodes * p;
    if (idx >= total) return;

    int node = idx / p;
    int d = idx - node * p;
    int64_t base = ((int64_t)b * max_nodes + node) * p + d;
    // CUDA 13 toolchains may not expose CUDART_INF_F in all compilation modes.
    // INFINITY is available via <cmath> and is valid in both host and device code.
    node_min[base] = INFINITY;
    node_max[base] = -INFINITY;
    node_sum[base] = 0.0f;
    if (d == 0) {
        node_count[(int64_t)b * max_nodes + node] = 0;
    }
}

__global__ void kdtree_accum_stats_kernel(
    const float* __restrict__ coords, // [B,N,p]
    const int32_t* __restrict__ node_id, // [B,N]
    const uint8_t* __restrict__ invalid_mask, // [B,N] (1=invalid) or nullptr
    int B, int N, int p,
    int nodes,
    int max_nodes,
    float* __restrict__ node_min, // [B,max_nodes,p]
    float* __restrict__ node_max, // [B,max_nodes,p]
    float* __restrict__ node_sum, // [B,max_nodes,p]
    int32_t* __restrict__ node_count // [B,max_nodes]
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || i >= N) return;
    if (invalid_mask && invalid_mask[(int64_t)b * N + i]) return;

    int32_t n = node_id[(int64_t)b * N + i];
    if ((int)n < 0 || (int)n >= nodes) return;

    atomicAdd(&node_count[(int64_t)b * max_nodes + n], 1);

    const float* ci = coords + ((int64_t)b * N + i) * p;
    float* bmin = node_min + (((int64_t)b * max_nodes + n) * p);
    float* bmax = node_max + (((int64_t)b * max_nodes + n) * p);
    float* bsum = node_sum + (((int64_t)b * max_nodes + n) * p);
    for (int d = 0; d < p; ++d) {
        float v = ci[d];
        atomicMinFloat(&bmin[d], v);
        atomicMaxFloat(&bmax[d], v);
        atomicAdd(&bsum[d], v);
    }
}

__global__ void kdtree_compute_splits_kernel(
    int B,
    int nodes,
    int max_nodes,
    int p,
    const float* __restrict__ node_min, // [B,max_nodes,p]
    const float* __restrict__ node_max, // [B,max_nodes,p]
    const float* __restrict__ node_sum, // [B,max_nodes,p]
    const int32_t* __restrict__ node_count, // [B,max_nodes]
    int32_t* __restrict__ split_dim, // [B,max_nodes]
    float* __restrict__ split_val  // [B,max_nodes]
) {
    int b = blockIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || n >= nodes) return;

    int64_t base_n = (int64_t)b * max_nodes + n;
    int32_t cnt = node_count[base_n];
    if (cnt <= 0) {
        split_dim[base_n] = 0;
        split_val[base_n] = 0.0f;
        return;
    }

    const float* bmin = node_min + base_n * p;
    const float* bmax = node_max + base_n * p;
    const float* bsum = node_sum + base_n * p;

    int best_dim = 0;
    float best_range = bmax[0] - bmin[0];
    for (int d = 1; d < p; ++d) {
        float r = bmax[d] - bmin[d];
        if (r > best_range) { best_range = r; best_dim = d; }
    }

    split_dim[base_n] = (int32_t)best_dim;
    split_val[base_n] = bsum[best_dim] / (float)cnt;
}

__global__ void kdtree_update_node_id_kernel(
    const float* __restrict__ coords, // [B,N,p]
    const int32_t* __restrict__ split_dim, // [B,max_nodes]
    const float* __restrict__ split_val, // [B,max_nodes]
    const uint8_t* __restrict__ invalid_mask, // [B,N] (1=invalid) or nullptr
    int B, int N, int p,
    int nodes,
    int max_nodes,
    int32_t* __restrict__ node_id // [B,N] in/out
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || i >= N) return;
    if (invalid_mask && invalid_mask[(int64_t)b * N + i]) return;

    int32_t n = node_id[(int64_t)b * N + i];
    if ((int)n < 0 || (int)n >= nodes) n = 0;

    int64_t base_n = (int64_t)b * max_nodes + n;
    int32_t dim = split_dim[base_n];
    dim = (dim < 0) ? 0 : (dim >= p ? (p - 1) : dim);
    float thr = split_val[base_n];

    const float* ci = coords + ((int64_t)b * N + i) * p;
    float v = ci[dim];

    // Next-level node id in [0, 2*nodes).
    int32_t nn = (v <= thr) ? (2 * n) : (2 * n + 1);
    node_id[(int64_t)b * N + i] = nn;
}

__global__ void kdtree_finalize_bucket_id_kernel(
    const int32_t* __restrict__ node_id, // [B,N]
    const uint8_t* __restrict__ invalid_mask, // [B,N] (1=invalid) or nullptr
    int B, int N,
    int key_range,
    int32_t* __restrict__ bucket_id // [B,N]
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || i >= N) return;
    if (invalid_mask && invalid_mask[(int64_t)b * N + i]) {
        bucket_id[(int64_t)b * N + i] = -1;
        return;
    }
    int32_t leaf = node_id[(int64_t)b * N + i];
    if (leaf < 0) leaf = 0;
    if (leaf >= key_range) leaf = key_range - 1;
    bucket_id[(int64_t)b * N + i] = leaf;
}

__global__ void bbox_counts_kernel(
    const float* __restrict__ coords1, // [B,N,p] float32
    const float* __restrict__ coords2, // [B,N,p] float32 (optional; may be nullptr)
    const int32_t* __restrict__ bucket_id, // [B,N]
    int B, int N, int p, int key_range,
    float* __restrict__ bbox1_min, // [B,key_range,p]
    float* __restrict__ bbox1_max, // [B,key_range,p]
    float* __restrict__ bbox2_min, // [B,key_range,p] (optional; may be nullptr)
    float* __restrict__ bbox2_max, // [B,key_range,p] (optional; may be nullptr)
    int32_t* __restrict__ bucket_count // [B,key_range]
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || i >= N) return;

    int32_t key = bucket_id[b * (int64_t)N + i];
    if ((int)key < 0 || (int)key >= key_range) return;

    atomicAdd(&bucket_count[b * key_range + key], 1);

    const int64_t leaf_base = ((int64_t)b * key_range + key) * p;

    // bbox in projection 1
    float* b1min = bbox1_min + leaf_base;
    float* b1max = bbox1_max + leaf_base;
    const float* c1 = coords1 + ((int64_t)b * N + i) * p;
    for (int d = 0; d < p; d++) {
        float v = c1[d];
        atomicMinFloat(&b1min[d], v);
        atomicMaxFloat(&b1max[d], v);
    }

    // bbox in projection 2 (optional) – used for tighter pruning bounds.
    if (coords2 != nullptr && bbox2_min != nullptr && bbox2_max != nullptr) {
        float* b2min = bbox2_min + leaf_base;
        float* b2max = bbox2_max + leaf_base;
        const float* c2 = coords2 + ((int64_t)b * N + i) * p;
        for (int d = 0; d < p; d++) {
            float v = c2[d];
            atomicMinFloat(&b2min[d], v);
            atomicMaxFloat(&b2max[d], v);
        }
    }
}



// -------------------------
// Bucket CSR construction (one-time per call)
// -------------------------

// Compute exclusive offsets from bucket_count.
// bucket_offsets has shape [B, key_range + 1] and bucket_offsets[..., 0] = 0.
__global__ void build_bucket_offsets_kernel(
    const int32_t* __restrict__ bucket_count, // [B,key_range]
    int B, int key_range,
    int32_t* __restrict__ bucket_offsets // [B,key_range+1]
) {
    int b = blockIdx.x;
    if (b >= B) return;
    if (threadIdx.x != 0) return;

    int64_t base_c = (int64_t)b * key_range;
    int64_t base_o = (int64_t)b * (key_range + 1);
    bucket_offsets[base_o + 0] = 0;
    int32_t running = 0;
    for (int j = 0; j < key_range; ++j) {
        int32_t c = bucket_count[base_c + j];
        // c should be >=0; clamp defensively.
        if (c < 0) c = 0;
        running += c;
        bucket_offsets[base_o + (j + 1)] = running;
    }
}

// Fill bucket_indices so that points for each leaf bucket are contiguous.
// Uses atomic cursors per bucket.
__global__ void fill_bucket_indices_kernel(
    const int32_t* __restrict__ bucket_id,    // [B,N]
    const int32_t* __restrict__ bucket_offsets, // [B,key_range+1]
    int32_t* __restrict__ bucket_cursor,      // [B,key_range] (initialized to 0)
    int B, int N, int key_range,
    int32_t* __restrict__ bucket_indices      // [B,N]
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || i >= N) return;

    int32_t key = bucket_id[(int64_t)b * N + i];
    if ((int)key < 0 || (int)key >= key_range) return;

    // Reserve a slot in this bucket.
    int32_t pos = atomicAdd(&bucket_cursor[(int64_t)b * key_range + key], 1);
    int32_t dst = bucket_offsets[(int64_t)b * (key_range + 1) + key] + pos;
    if ((int)dst < 0 || (int)dst >= N) return;
    bucket_indices[(int64_t)b * N + dst] = i;
}

// -------------------------
// FPS state kernels
// -------------------------

template <typename scalar_t>
__global__ void init_best_and_mindist_kernel(
    const scalar_t* __restrict__ x, // [B,N,D]
    const int32_t* __restrict__ bucket_id, // [B,N]
    const int64_t* __restrict__ start_idx, // [B]
    uint8_t* __restrict__ selected_mask, // [B,N] (1=not eligible)
    int B, int N, int D, int key_range,
    float* __restrict__ min_dist, // [B,N]
    int64_t* __restrict__ bucket_best_key // [B,key_range]
) {
    int b = blockIdx.y;
    if (b >= B) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool in_range = (i < N);

    const scalar_t* xb = x + ((int64_t)b * N * D);
    int64_t sidx = start_idx[b];
    sidx = (sidx < 0) ? 0 : ((sidx >= N) ? (N - 1) : sidx);

    // Mark the start point as selected (not eligible).
    if (in_range && (int64_t)i == sidx) {
        selected_mask[(int64_t)b * N + i] = 1;
    }

    extern __shared__ float sh_ref[];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        sh_ref[d] = to_float(xb[(int64_t)sidx * D + d]);
    }
    __syncthreads();

    if (!in_range) return;

    float dist = 0.0f;
    const scalar_t* xi = xb + (int64_t)i * D;
    for (int d = 0; d < D; d++) {
        float dv = to_float(xi[d]) - sh_ref[d];
        dist += dv * dv;
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
    uint8_t* __restrict__ selected_mask, // [B,N]
    int B, int N, int key_range,
    int64_t* __restrict__ out_idx, // [B]
    int64_t* __restrict__ out_indices, // [B,k] (optional, may be nullptr)
    int out_stride_k,
    int out_col
) {
    int b = blockIdx.x;
    if (b >= B) return;

    __shared__ unsigned long long sh_best_k[256];
    int t = threadIdx.x;
    unsigned long long best_k = block_reduce_best_key(
        bucket_best_key, bucket_count, selected_mask, B, N, key_range, b, sh_best_k);

    if (t == 0) {
        int64_t chosen = 0;

        if (best_k == 0ull && selected_mask) {
            // Fallback: pick the first unselected index (degenerate cases / stale bucket bests).
            for (int64_t i = 0; i < (int64_t)N; ++i) {
                if (!selected_mask[(int64_t)b * N + i]) { chosen = i; break; }
            }
        } else {
            chosen = (int64_t)unpack_best_idx(best_k);
        }

        out_idx[b] = chosen;
        if (out_indices) {
            out_indices[(int64_t)b * out_stride_k + out_col] = chosen;
        }

        // Fuse mark-selected here.
        if (selected_mask) {
            int64_t ci = (chosen < 0) ? 0 : ((chosen >= N) ? (N - 1) : chosen);
            selected_mask[(int64_t)b * N + ci] = 1;
        }
    }
}

// Fused reduce + active mask kernel to cut one launch per iteration.
// Computes next index and active buckets in one pass (per batch block).
__global__ void reduce_and_active_mask_kernel(
    const float* __restrict__ coords1, // [B,N,p] float32
    const float* __restrict__ coords2, // [B,N,p] (optional; may be nullptr)
    const int32_t* __restrict__ bucket_id, // [B,N]
    const float* __restrict__ bbox1_min, // [B,key_range,p]
    const float* __restrict__ bbox1_max, // [B,key_range,p]
    const float* __restrict__ bbox2_min, // [B,key_range,p] (optional; may be nullptr)
    const float* __restrict__ bbox2_max, // [B,key_range,p] (optional; may be nullptr)
    int64_t* __restrict__ bucket_best_key, // [B,key_range]
    const int32_t* __restrict__ bucket_count, // [B,key_range]
    uint8_t* __restrict__ selected_mask, // [B,N]
    int B, int N, int p, int key_range,
    int64_t* __restrict__ out_idx, // [B]
    int64_t* __restrict__ out_indices, // [B,k] (optional, may be nullptr)
    int out_stride_k,
    int out_col,
    uint8_t* __restrict__ active_mask // [B,key_range]
) {
    int b = blockIdx.x;
    if (b >= B) return;

    __shared__ unsigned long long sh_best_k[256];
    int t = threadIdx.x;
    unsigned long long best_k = block_reduce_best_key(
        bucket_best_key, bucket_count, selected_mask, B, N, key_range, b, sh_best_k);

    __shared__ int64_t sh_chosen;
    if (t == 0) {
        int64_t chosen = 0;

        if (best_k == 0ull && selected_mask) {
            // Fallback: pick the first unselected index.
            for (int64_t i = 0; i < (int64_t)N; ++i) {
                if (!selected_mask[(int64_t)b * N + i]) { chosen = i; break; }
            }
        } else {
            chosen = (int64_t)unpack_best_idx(best_k);
        }

        out_idx[b] = chosen;
        if (out_indices) {
            out_indices[(int64_t)b * out_stride_k + out_col] = chosen;
        }

        // Mark selected (not eligible).
        if (selected_mask) {
            int64_t ci = (chosen < 0) ? 0 : ((chosen >= N) ? (N - 1) : chosen);
            selected_mask[(int64_t)b * N + ci] = 1;
        }
        sh_chosen = chosen;
    }
    __syncthreads();

    int64_t ridx = sh_chosen;
    ridx = (ridx < 0) ? 0 : ((ridx >= N) ? (N - 1) : ridx);

    int32_t ref_bucket = bucket_id[(int64_t)b * N + ridx];
    const float* cref1 = coords1 + ((int64_t)b * N + ridx) * p;
    const float* cref2 = (coords2 != nullptr) ? (coords2 + ((int64_t)b * N + ridx) * p) : nullptr;
    bool use_coord2 = (coords2 != nullptr && bbox2_min != nullptr && bbox2_max != nullptr);

    for (int j = threadIdx.x; j < key_range; j += blockDim.x) {
        int32_t cnt = bucket_count[b * key_range + j];
        if (cnt <= 0) {
            active_mask[b * (int64_t)key_range + j] = 0;
            continue;
        }

        unsigned long long key = (unsigned long long)bucket_best_key[b * (int64_t)key_range + j];
        float lastmax = unpack_best_dist(key);

        // Fast reject: if lastmax==0, nothing in this bucket can improve.
        if (!(lastmax > 0.0f)) {
            uint8_t active = (ref_bucket == j) ? 1 : 0;
            active_mask[b * (int64_t)key_range + j] = active;
            if (active) bucket_best_key[b * (int64_t)key_range + j] = 0;
            continue;
        }

        const int64_t leaf_base = ((int64_t)b * key_range + j) * p;
        const float* b1min = bbox1_min + leaf_base;
        const float* b1max = bbox1_max + leaf_base;

        float bound = 0.0f;
        for (int d = 0; d < p; ++d) {
            float v = cref1[d];
            float dd = 0.0f;
            float lo = b1min[d], hi = b1max[d];
            if (v > hi) dd = v - hi;
            else if (v < lo) dd = lo - v;
            bound += dd * dd;
        }

        if (use_coord2) {
            const float* b2min = bbox2_min + leaf_base;
            const float* b2max = bbox2_max + leaf_base;

            float bound2 = 0.0f;
            for (int d = 0; d < p; ++d) {
                float v = cref2[d];
                float dd = 0.0f;
                float lo = b2min[d], hi = b2max[d];
                if (v > hi) dd = v - hi;
                else if (v < lo) dd = lo - v;
                bound2 += dd * dd;
            }
            bound = fmaxf(bound, bound2);
        }

        uint8_t active = (bound < lastmax) ? 1 : 0;
        if (ref_bucket == j) active = 1;
        active_mask[b * (int64_t)key_range + j] = active;
        if (active) {
            bucket_best_key[b * (int64_t)key_range + j] = 0;
        }
    }
}

// Warp reduction for uint64 max.
__device__ __forceinline__ unsigned long long warp_reduce_max_u64(unsigned long long v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        unsigned long long other = __shfl_down_sync(0xffffffffu, v, offset);
        if (other > v) v = other;
    }
    return v;
}

// Warp reduction for float sum.
__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

template <typename scalar_t>
struct WarpL2;

// Generic fallback (no vectorization). Uses warp-coalesced strided loads.
template <typename scalar_t>
struct WarpL2 {
    __device__ __forceinline__ static float compute(const scalar_t* __restrict__ xi,
                                                    const scalar_t* __restrict__ ref,
                                                    int D,
                                                    float early_stop) {
        int lane = threadIdx.x & 31;
        float acc = 0.0f;

        // Check early-stop every 128 dims.
        constexpr int CHECK = 128;
        for (int base = 0; base < D; base += CHECK) {
            int end = min(D, base + CHECK);
            for (int d = base + lane; d < end; d += 32) {
                float dv = to_float(xi[d]) - to_float(ref[d]);
                acc += dv * dv;
            }
            float total = warp_reduce_sum_f32(acc);
            total = __shfl_sync(0xffffffffu, total, 0);
            if (total >= early_stop) return early_stop;
        }
        float total = warp_reduce_sum_f32(acc);
        return __shfl_sync(0xffffffffu, total, 0);
    }
};

template <>
struct WarpL2<float> {
    __device__ __forceinline__ static float compute(const float* __restrict__ xi,
                                                    const float* __restrict__ ref,
                                                    int D,
                                                    float early_stop) {
        // Process 4 dims per lane per step (float4), so one warp step covers 128 dims.
        int lane = threadIdx.x & 31;
        float acc = 0.0f;
        int D4 = D >> 2;          // floor(D/4)
        int tail = D & 3;         // D % 4

        const float4* xi4 = reinterpret_cast<const float4*>(xi);
        const float4* rf4 = reinterpret_cast<const float4*>(ref);

        // Early-exit check every 128 dims (one warp step).
        for (int base = 0; base < D4; base += 32) {
            int j = base + lane;
            if (j < D4) {
                float4 a = xi4[j];
                float4 b = rf4[j];
                float dv0 = a.x - b.x; acc += dv0 * dv0;
                float dv1 = a.y - b.y; acc += dv1 * dv1;
                float dv2 = a.z - b.z; acc += dv2 * dv2;
                float dv3 = a.w - b.w; acc += dv3 * dv3;
            }
            float total = warp_reduce_sum_f32(acc);
            total = __shfl_sync(0xffffffffu, total, 0);
            if (total >= early_stop) {
                // All lanes break consistently.
                return early_stop;
            }
        }

        // Handle tail dims (at most 3) without warp divergence.
        if (tail) {
            int start = D4 << 2;
            if (lane < tail) {
                int d = start + lane;
                float dv = xi[d] - ref[d];
                acc += dv * dv;
            }
        }
        float total = warp_reduce_sum_f32(acc);
        return __shfl_sync(0xffffffffu, total, 0);
    }
};

template <>
struct WarpL2<at::Half> {
    __device__ __forceinline__ static float compute(const at::Half* __restrict__ xi,
                                                    const at::Half* __restrict__ ref,
                                                    int D,
                                                    float early_stop) {
        int lane = threadIdx.x & 31;
        float acc = 0.0f;
        int D2 = D >> 1;        // floor(D/2)
        int tail = D & 1;       // D % 2
        const __half2* xi2 = reinterpret_cast<const __half2*>(xi);
        const __half2* rf2 = reinterpret_cast<const __half2*>(ref);

        // One warp step loads one half2 per lane => 64 dims per step.
        // Check early-stop every 2 steps => 128 dims.
        int step = 0;
        for (int base = 0; base < D2; base += 32) {
            int j = base + lane;
            if (j < D2) {
                __half2 a2 = xi2[j];
                __half2 b2 = rf2[j];
                float2 af = __half22float2(a2);
                float2 bf = __half22float2(b2);
                float dv0 = af.x - bf.x; acc += dv0 * dv0;
                float dv1 = af.y - bf.y; acc += dv1 * dv1;
            }
            step++;
            if (step == 2 || (base + 32) >= D2) {
                float total = warp_reduce_sum_f32(acc);
                total = __shfl_sync(0xffffffffu, total, 0);
                if (total >= early_stop) return early_stop;
                step = 0;
            }
        }

        // Tail dim (at most 1).
        if (tail && (lane == 0)) {
            int d = D2 << 1;
            float dv = __half2float(reinterpret_cast<const __half*>(xi)[d]) -
                       __half2float(reinterpret_cast<const __half*>(ref)[d]);
            acc += dv * dv;
        }
        __syncwarp();
        // Tail could push us over early_stop.
        if (__shfl_sync(0xffffffffu, acc, 0) >= early_stop) {
            // Not exact (acc is lane-local), so we do a real check below.
        }
        {
            float total = warp_reduce_sum_f32(acc);
            total = __shfl_sync(0xffffffffu, total, 0);
            if (total >= early_stop) return early_stop;
        }
        float total = warp_reduce_sum_f32(acc);
        return __shfl_sync(0xffffffffu, total, 0);
    }
};

template <>
struct WarpL2<at::BFloat16> {
    __device__ __forceinline__ static float compute(const at::BFloat16* __restrict__ xi,
                                                    const at::BFloat16* __restrict__ ref,
                                                    int D,
                                                    float early_stop) {
        int lane = threadIdx.x & 31;
        float acc = 0.0f;
        int D2 = D >> 1;
        int tail = D & 1;
        const __nv_bfloat162* xi2 = reinterpret_cast<const __nv_bfloat162*>(xi);
        const __nv_bfloat162* rf2 = reinterpret_cast<const __nv_bfloat162*>(ref);

        int step = 0;
        for (int base = 0; base < D2; base += 32) {
            int j = base + lane;
            if (j < D2) {
                __nv_bfloat162 a2 = xi2[j];
                __nv_bfloat162 b2 = rf2[j];
                float2 af = __bfloat1622float2(a2);
                float2 bf = __bfloat1622float2(b2);
                float dv0 = af.x - bf.x; acc += dv0 * dv0;
                float dv1 = af.y - bf.y; acc += dv1 * dv1;
            }
            step++;
            if (step == 2 || (base + 32) >= D2) {
                float total = warp_reduce_sum_f32(acc);
                total = __shfl_sync(0xffffffffu, total, 0);
                if (total >= early_stop) return early_stop;
                step = 0;
            }
        }

        if (tail && (lane == 0)) {
            int d = D2 << 1;
            float dv = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(xi)[d]) -
                       __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ref)[d]);
            acc += dv * dv;
        }
        __syncwarp();
        {
            float total = warp_reduce_sum_f32(acc);
            total = __shfl_sync(0xffffffffu, total, 0);
            if (total >= early_stop) return early_stop;
        }
        float total = warp_reduce_sum_f32(acc);
        return __shfl_sync(0xffffffffu, total, 0);
    }
};

template <typename scalar_t, bool USE_COORD2>
__global__ void update_leaves_kernel(
    const scalar_t* __restrict__ x, // [B,N,D]
    const float* __restrict__ coords1, // [B,N,p]
    const float* __restrict__ coords2, // [B,N,p] (optional; may be nullptr if !USE_COORD2)
    int p,
    const int32_t* __restrict__ bucket_offsets, // [B,key_range+1]
    const int32_t* __restrict__ bucket_indices, // [B,N]
    const uint8_t* __restrict__ active_mask, // [B,key_range]
    const uint8_t* __restrict__ selected_mask, // [B,N]
    const int64_t* __restrict__ ref_idx, // [B]
    int B, int N, int D, int key_range,
    float* __restrict__ min_dist, // [B,N]
    int64_t* __restrict__ bucket_best_key // [B,key_range]
) {
    int leaf = blockIdx.x;
    int b = blockIdx.y;
    if (b >= B || leaf >= key_range) return;

    if (!active_mask[(int64_t)b * key_range + leaf]) return;

    int64_t ridx = ref_idx[b];
    ridx = (ridx < 0) ? 0 : ((ridx >= N) ? (N - 1) : ridx);

    const scalar_t* xb = x + ((int64_t)b * N * D);
    const scalar_t* xref = xb + (int64_t)ridx * D;

    // Cache the reference vector in shared memory (saves repeated global loads).
    extern __shared__ __align__(16) unsigned char smem[];
    scalar_t* sh_ref = reinterpret_cast<scalar_t*>(smem);
    for (int d = threadIdx.x; d < D; d += (int)blockDim.x) {
        sh_ref[d] = xref[d];
    }
    __syncthreads();

    const float* cref1 = coords1 + ((int64_t)b * N + ridx) * p;
    const float* cref2 = USE_COORD2 ? (coords2 + ((int64_t)b * N + ridx) * p) : nullptr;

    int64_t base_o = (int64_t)b * (key_range + 1);
    int32_t start = bucket_offsets[base_o + leaf];
    int32_t end   = bucket_offsets[base_o + (leaf + 1)];
    if (start < 0) start = 0;
    if (end > N) end = N;
    if (end <= start) {
        bucket_best_key[(int64_t)b * key_range + leaf] = 0;
        return;
    }

    unsigned long long local_best = 0ull;

    // Distance compute: projection lower-bound + warp-coalesced full-D distance.
    // We use one warp per candidate point, coalescing loads across lanes.
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    int num_warps = (int)blockDim.x >> 5;

    for (int32_t t = start + wid; t < end; t += num_warps) {
        int32_t i = -1;
        if (lane == 0) {
            i = bucket_indices[(int64_t)b * N + t];
        }
        i = __shfl_sync(0xffffffffu, i, 0);
        if ((int)i < 0 || (int)i >= N) continue;

        int skip = 0;
        if (lane == 0) {
            if (selected_mask && selected_mask[(int64_t)b * N + i]) skip = 1;
        }
        skip = __shfl_sync(0xffffffffu, skip, 0);
        if (skip) continue;

        float old = 0.0f;
        if (lane == 0) {
            old = min_dist[(int64_t)b * N + i];
        }
        old = __shfl_sync(0xffffffffu, old, 0);

        // Projection-space bound (done on lane 0; p <= 64).
        float lb = 0.0f;
        if (lane == 0) {
            const float* ci1 = coords1 + ((int64_t)b * N + i) * p;
            #pragma unroll
            for (int dd = 0; dd < kMaxBucketingP; ++dd) {
                if (dd >= p) break;
                float dv = ci1[dd] - cref1[dd];
                lb += dv * dv;
            }
            if constexpr (USE_COORD2) {
                float lb2 = 0.0f;
                const float* ci2 = coords2 + ((int64_t)b * N + i) * p;
                #pragma unroll
                for (int dd = 0; dd < kMaxBucketingP; ++dd) {
                    if (dd >= p) break;
                    float dv = ci2[dd] - cref2[dd];
                    lb2 += dv * dv;
                }
                lb = fmaxf(lb, lb2);
            }
        }
        lb = __shfl_sync(0xffffffffu, lb, 0);

        if (lb >= old) {
            if (lane == 0) {
                unsigned long long k = pack_best_key(old, i);
                if (k > local_best) local_best = k;
            }
            continue;
        }

        const scalar_t* xi = xb + (int64_t)i * D;
        float dist = WarpL2<scalar_t>::compute(xi, sh_ref, D, old);

        if (lane == 0) {
            float nd = dist;
            if (nd < old) {
                min_dist[(int64_t)b * N + i] = nd;
            } else {
                nd = old;
            }
            unsigned long long k = pack_best_key(nd, i);
            if (k > local_best) local_best = k;
        }
    }

    // Block-wide max reduction of the packed (dist,idx) key.
    unsigned long long v = warp_reduce_max_u64(local_best);
    __shared__ unsigned long long warp_best[32];
    if (lane == 0) warp_best[wid] = v;
    __syncthreads();

    unsigned long long block_best = 0ull;
    if (wid == 0) {
        block_best = (threadIdx.x < (blockDim.x >> 5)) ? warp_best[lane] : 0ull;
        block_best = warp_reduce_max_u64(block_best);
    }

    if (threadIdx.x == 0) {
        bucket_best_key[(int64_t)b * key_range + leaf] = (int64_t)block_best;
    }
}




// ----
// ABI-stable host entrypoints
// ----

// The original kernels were written against c10::Half / c10::BFloat16.
// Keep kernel scalar_t instantiations in terms of those wrapper types.
// We provide explicit specializations for safe float conversion.

template <>
__device__ __forceinline__ float to_float<at::Half>(at::Half v) {
    // at::Half / c10::Half is layout-compatible with CUDA __half
    return __half2float(*reinterpret_cast<const __half*>(&v));
}

template <>
__device__ __forceinline__ float to_float<at::BFloat16>(at::BFloat16 v) {
    // at::BFloat16 / c10::BFloat16 is layout-compatible with CUDA __nv_bfloat16
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&v));
}

// These specializations are kept for completeness (some device helpers may
// use CUDA native types directly), but the main kernels instantiate on
// at::Half / at::BFloat16.
template <>
__device__ __forceinline__ float to_float<__half>(__half v) {
    return __half2float(v);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

__global__ void write_first_col_kernel(
    const int64_t* __restrict__ start_idx,
    int64_t* __restrict__ out_indices,
    int B,
    int k
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    out_indices[(int64_t)b * (int64_t)k] = start_idx[b];
}

inline cudaStream_t _get_current_stream_or_throw(int device_index) {
    void* stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(
        aoti_torch_get_current_cuda_stream(device_index, &stream_ptr));
    return static_cast<cudaStream_t>(stream_ptr);
}

inline void _check_cuda_tensor(const torch::stable::Tensor& t, const char* name) {
    STD_TORCH_CHECK(t.device().type() == torch::headeronly::DeviceType::CUDA,
                    name, " must be a CUDA tensor");
}

inline void _check_same_device(const torch::stable::Tensor& a, const torch::stable::Tensor& b, const char* an, const char* bn) {
    STD_TORCH_CHECK(a.get_device_index() == b.get_device_index(), an, " and ", bn, " must be on the same CUDA device");
}

static torch::stable::Tensor sample_idx_cuda_bucket_impl(
    const torch::stable::Tensor& x,
    int64_t k,
    int64_t h,
    const torch::stable::Tensor& start_idx,
    const torch::stable::Tensor& invalid_mask,
    int64_t low_d
) {
    _check_cuda_tensor(x, "x");
    _check_cuda_tensor(start_idx, "start_idx");
    _check_cuda_tensor(invalid_mask, "invalid_mask");
    _check_same_device(x, start_idx, "x", "start_idx");
    _check_same_device(x, invalid_mask, "x", "invalid_mask");

    STD_TORCH_CHECK(x.dim() == 3, "x must be 3D [B,N,D], got dim=", x.dim());
    STD_TORCH_CHECK(k >= 1, "k must be >= 1, got ", k);

    const auto st = x.scalar_type();
    STD_TORCH_CHECK(
        st == torch::headeronly::ScalarType::Float ||
        st == torch::headeronly::ScalarType::Half ||
        st == torch::headeronly::ScalarType::BFloat16,
        "x must have dtype float32/float16/bfloat16 on CUDA");

    STD_TORCH_CHECK(start_idx.scalar_type() == torch::headeronly::ScalarType::Long,
                    "start_idx must have dtype int64");
    STD_TORCH_CHECK(invalid_mask.scalar_type() == torch::headeronly::ScalarType::Byte,
                    "invalid_mask must have dtype uint8");

    auto x_contig = torch::stable::contiguous(x);

    const int64_t B64 = x_contig.size(0);
    const int64_t N64 = x_contig.size(1);
    const int64_t D64 = x_contig.size(2);

    STD_TORCH_CHECK(start_idx.dim() == 1 && start_idx.size(0) == B64,
                    "start_idx must have shape [B]");
    STD_TORCH_CHECK(invalid_mask.dim() == 2 && invalid_mask.size(0) == B64 && invalid_mask.size(1) == N64,
                    "invalid_mask must have shape [B,N]");

    STD_TORCH_CHECK(k <= N64, "k must be <= N. Got k=", k, " N=", N64);

    int64_t height = std::max<int64_t>(1, h);
    STD_TORCH_CHECK(height <= 12,
                    "h (KD-tree height) must be <= 12 for CUDA (max 4096 leaves), got ", height);
    const int key_range = 1 << (int)height;

    const int D = (int)D64;
    const int B = (int)B64;
    const int N = (int)N64;

    // Bucketing dimension.
    int p = (low_d > 0) ? (int)std::min<int64_t>((int64_t)D, low_d) : D;
    STD_TORCH_CHECK(p >= 1, "p must be >= 1, got ", p);
    STD_TORCH_CHECK(p <= kMaxBucketingP,
                    "For CUDA KD-tree bucketing, p must be <= ", kMaxBucketingP,
                    ". Got p=", p, " (D=", D, "). Please set low_d<=", kMaxBucketingP, ".");

    // Make sure we are on the right device.
    CUDA_CHECK(cudaSetDevice(x.get_device_index()));
    cudaStream_t stream = _get_current_stream_or_throw(x.get_device_index());

    const int threads = 256;
    dim3 block(threads);
    dim3 grid_points((N + threads - 1) / threads, B);

    // selected_mask: 1 means "not eligible" (invalid or already selected).
    auto invalid_c = torch::stable::contiguous(invalid_mask);
    torch::stable::Tensor selected_mask = torch::stable::new_empty(
        invalid_c, {B64, N64}, torch::headeronly::ScalarType::Byte);
    CUDA_CHECK(cudaMemcpyAsync(
        selected_mask.mutable_data_ptr<uint8_t>(),
        invalid_c.const_data_ptr<uint8_t>(),
        (size_t)B64 * (size_t)N64 * sizeof(uint8_t),
        cudaMemcpyDeviceToDevice,
        stream));

    // Build coordinates used for KD-tree bucketing + pruning.
    const bool use_coords2 = (p != D);

    // Initialize to a valid (possibly empty) tensor so we don't rely on a default constructor.
    torch::stable::Tensor coords_f32  = torch::stable::new_empty(x_contig, {0}, torch::headeronly::ScalarType::Float);
    torch::stable::Tensor coords2_f32 = torch::stable::new_empty(x_contig, {0}, torch::headeronly::ScalarType::Float);

    if (!use_coords2) {
        coords_f32 = torch::stable::contiguous(torch::stable::to(x_contig, torch::headeronly::ScalarType::Float));
    } else {
        coords_f32  = torch::stable::new_empty(x_contig, {B64, N64, (int64_t)p}, torch::headeronly::ScalarType::Float);
        coords2_f32 = torch::stable::new_empty(x_contig, {B64, N64, (int64_t)p}, torch::headeronly::ScalarType::Float);

        const uint32_t seed1 = 0x6d2b79f5u ^ (uint32_t)D * 0x9e3779b9u ^ (uint32_t)p * 0x85ebca6bu;
        const uint32_t seed2 = seed1 ^ 0x68bc21ebu;

        float* c1 = coords_f32.mutable_data_ptr<float>();
        float* c2 = coords2_f32.mutable_data_ptr<float>();

        if (st == torch::headeronly::ScalarType::Float) {
            const float* xptr = x_contig.const_data_ptr<float>();
            project_rademacher_kernel<float><<<grid_points, block, 0, stream>>>(xptr, B, N, D, p, seed1, c1);
            project_rademacher_kernel<float><<<grid_points, block, 0, stream>>>(xptr, B, N, D, p, seed2, c2);
        } else if (st == torch::headeronly::ScalarType::Half) {
            // NOTE: torch::stable::Tensor only guarantees const_data_ptr<T>() for
            // PyTorch scalar wrapper types (e.g., at::Half / at::BFloat16), not
            // CUDA builtin types (__half / __nv_bfloat16). Using CUDA builtin
            // types here will compile but can fail to link on some builds.
            const at::Half* xptr = x_contig.const_data_ptr<at::Half>();
            project_rademacher_kernel<at::Half><<<grid_points, block, 0, stream>>>(xptr, B, N, D, p, seed1, c1);
            project_rademacher_kernel<at::Half><<<grid_points, block, 0, stream>>>(xptr, B, N, D, p, seed2, c2);
        } else { // BFloat16
            const at::BFloat16* xptr = x_contig.const_data_ptr<at::BFloat16>();
            project_rademacher_kernel<at::BFloat16><<<grid_points, block, 0, stream>>>(xptr, B, N, D, p, seed1, c1);
            project_rademacher_kernel<at::BFloat16><<<grid_points, block, 0, stream>>>(xptr, B, N, D, p, seed2, c2);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // Allocate intermediate tensors.
    torch::stable::Tensor bucket_id = torch::stable::new_empty(x_contig, {B64, N64}, torch::headeronly::ScalarType::Int);
    torch::stable::fill_(bucket_id, -1.0);

    torch::stable::Tensor bbox_min = torch::stable::new_empty(x_contig, {B64, (int64_t)key_range, (int64_t)p}, torch::headeronly::ScalarType::Float);
    torch::stable::Tensor bbox_max = torch::stable::new_empty(x_contig, {B64, (int64_t)key_range, (int64_t)p}, torch::headeronly::ScalarType::Float);
    torch::stable::fill_(bbox_min, (double)INFINITY);
    torch::stable::fill_(bbox_max, (double)-INFINITY);

    // Optional second set of bounding boxes used only when p < D (Rademacher projections).
    torch::stable::Tensor bbox2_min = torch::stable::new_empty(x_contig, {0}, torch::headeronly::ScalarType::Float);
    torch::stable::Tensor bbox2_max = torch::stable::new_empty(x_contig, {0}, torch::headeronly::ScalarType::Float);
    if (use_coords2) {
        bbox2_min = torch::stable::new_empty(x_contig, {B64, (int64_t)key_range, (int64_t)p}, torch::headeronly::ScalarType::Float);
        bbox2_max = torch::stable::new_empty(x_contig, {B64, (int64_t)key_range, (int64_t)p}, torch::headeronly::ScalarType::Float);
        torch::stable::fill_(bbox2_min, (double)INFINITY);
        torch::stable::fill_(bbox2_max, (double)-INFINITY);
    }

    torch::stable::Tensor bucket_count = torch::stable::new_empty(x_contig, {B64, (int64_t)key_range}, torch::headeronly::ScalarType::Int);
    torch::stable::fill_(bucket_count, 0.0);

    torch::stable::Tensor bucket_offsets = torch::stable::new_empty(x_contig, {B64, (int64_t)key_range + 1}, torch::headeronly::ScalarType::Int);
    torch::stable::Tensor bucket_indices = torch::stable::new_empty(x_contig, {B64, N64}, torch::headeronly::ScalarType::Int);
    torch::stable::Tensor bucket_cursor  = torch::stable::new_empty(x_contig, {B64, (int64_t)key_range}, torch::headeronly::ScalarType::Int);
    torch::stable::fill_(bucket_cursor, 0.0);

    torch::stable::Tensor bucket_best_key = torch::stable::new_empty(x_contig, {B64, (int64_t)key_range}, torch::headeronly::ScalarType::Long);
    torch::stable::fill_(bucket_best_key, 0.0);

    torch::stable::Tensor active_mask = torch::stable::new_empty(x_contig, {B64, (int64_t)key_range}, torch::headeronly::ScalarType::Byte);
    torch::stable::fill_(active_mask, 0.0);

    torch::stable::Tensor min_dist = torch::stable::new_empty(x_contig, {B64, N64}, torch::headeronly::ScalarType::Float);

    torch::stable::Tensor out_indices = torch::stable::new_empty(x_contig, {B64, k}, torch::headeronly::ScalarType::Long);
    torch::stable::fill_(out_indices, -1.0);

    // Set out_indices[:, 0] = start_idx.
    {
        const int t = 128;
        const int blocks = (B + t - 1) / t;
        write_first_col_kernel<<<blocks, t, 0, stream>>>(
            start_idx.const_data_ptr<int64_t>(),
            out_indices.mutable_data_ptr<int64_t>(),
            B,
            (int)k);
        CUDA_CHECK(cudaGetLastError());
    }

    // ---- KD-tree bucketing (GPU build, complete tree of height `h`) ----
    const uint8_t* invalid_ptr = selected_mask.const_data_ptr<uint8_t>();

    torch::stable::Tensor node_id = torch::stable::new_empty(x_contig, {B64, N64}, torch::headeronly::ScalarType::Int);
    torch::stable::fill_(node_id, 0.0);

    torch::stable::Tensor node_min = torch::stable::new_empty(x_contig, {B64, (int64_t)key_range, (int64_t)p}, torch::headeronly::ScalarType::Float);
    torch::stable::Tensor node_max = torch::stable::new_empty(x_contig, {B64, (int64_t)key_range, (int64_t)p}, torch::headeronly::ScalarType::Float);
    torch::stable::Tensor node_sum = torch::stable::new_empty(x_contig, {B64, (int64_t)key_range, (int64_t)p}, torch::headeronly::ScalarType::Float);
    torch::stable::Tensor node_count = torch::stable::new_empty(x_contig, {B64, (int64_t)key_range}, torch::headeronly::ScalarType::Int);

    torch::stable::Tensor split_dim_lvl = torch::stable::new_empty(x_contig, {B64, (int64_t)key_range}, torch::headeronly::ScalarType::Int);
    torch::stable::Tensor split_val_lvl = torch::stable::new_empty(x_contig, {B64, (int64_t)key_range}, torch::headeronly::ScalarType::Float);

    for (int depth = 0; depth < (int)height; ++depth) {
        int nodes = 1 << depth;
        int total = nodes * p;
        dim3 grid_stats((total + threads - 1) / threads, B);

        kdtree_reset_stats_kernel<<<grid_stats, block, 0, stream>>>(
            B,
            key_range,
            nodes,
            p,
            node_min.mutable_data_ptr<float>(),
            node_max.mutable_data_ptr<float>(),
            node_sum.mutable_data_ptr<float>(),
            node_count.mutable_data_ptr<int32_t>());
        CUDA_CHECK(cudaGetLastError());

        kdtree_accum_stats_kernel<<<grid_points, block, 0, stream>>>(
            coords_f32.const_data_ptr<float>(),
            node_id.const_data_ptr<int32_t>(),
            invalid_ptr,
            B, N, p,
            nodes,
            key_range,
            node_min.mutable_data_ptr<float>(),
            node_max.mutable_data_ptr<float>(),
            node_sum.mutable_data_ptr<float>(),
            node_count.mutable_data_ptr<int32_t>());
        CUDA_CHECK(cudaGetLastError());

        dim3 grid_nodes(((nodes + threads - 1) / threads), B);
        kdtree_compute_splits_kernel<<<grid_nodes, block, 0, stream>>>(
            B,
            nodes,
            key_range,
            p,
            node_min.const_data_ptr<float>(),
            node_max.const_data_ptr<float>(),
            node_sum.const_data_ptr<float>(),
            node_count.const_data_ptr<int32_t>(),
            split_dim_lvl.mutable_data_ptr<int32_t>(),
            split_val_lvl.mutable_data_ptr<float>());
        CUDA_CHECK(cudaGetLastError());

        kdtree_update_node_id_kernel<<<grid_points, block, 0, stream>>>(
            coords_f32.const_data_ptr<float>(),
            split_dim_lvl.const_data_ptr<int32_t>(),
            split_val_lvl.const_data_ptr<float>(),
            invalid_ptr,
            B, N, p,
            nodes,
            key_range,
            node_id.mutable_data_ptr<int32_t>());
        CUDA_CHECK(cudaGetLastError());
    }

    kdtree_finalize_bucket_id_kernel<<<grid_points, block, 0, stream>>>(
        node_id.const_data_ptr<int32_t>(),
        invalid_ptr,
        B, N,
        key_range,
        bucket_id.mutable_data_ptr<int32_t>());
    CUDA_CHECK(cudaGetLastError());
    const float* coords2_ptr = use_coords2 ? coords2_f32.const_data_ptr<float>() : nullptr;
    float* bbox2_min_ptr = use_coords2 ? bbox2_min.mutable_data_ptr<float>() : nullptr;
    float* bbox2_max_ptr = use_coords2 ? bbox2_max.mutable_data_ptr<float>() : nullptr;

    bbox_counts_kernel<<<grid_points, block, 0, stream>>>(
        coords_f32.const_data_ptr<float>(),
        coords2_ptr,
        bucket_id.const_data_ptr<int32_t>(),
        B, N, p, key_range,
        bbox_min.mutable_data_ptr<float>(),
        bbox_max.mutable_data_ptr<float>(),
        bbox2_min_ptr,
        bbox2_max_ptr,
        bucket_count.mutable_data_ptr<int32_t>());
    CUDA_CHECK(cudaGetLastError());

    build_bucket_offsets_kernel<<<B, 1, 0, stream>>>(
        bucket_count.const_data_ptr<int32_t>(),
        B, key_range,
        bucket_offsets.mutable_data_ptr<int32_t>());
    CUDA_CHECK(cudaGetLastError());

    fill_bucket_indices_kernel<<<grid_points, block, 0, stream>>>(
        bucket_id.const_data_ptr<int32_t>(),
        bucket_offsets.const_data_ptr<int32_t>(),
        bucket_cursor.mutable_data_ptr<int32_t>(),
        B, N, key_range,
        bucket_indices.mutable_data_ptr<int32_t>());
    CUDA_CHECK(cudaGetLastError());

    // ---- FPS ----
    const size_t shmem_init = (size_t)D * sizeof(float);

    // Init step.
    if (st == torch::headeronly::ScalarType::Float) {
        const float* xptr = x_contig.const_data_ptr<float>();
        init_best_and_mindist_kernel<float><<<grid_points, block, shmem_init, stream>>>(
            xptr,
            bucket_id.const_data_ptr<int32_t>(),
            start_idx.const_data_ptr<int64_t>(),
            selected_mask.mutable_data_ptr<uint8_t>(),
            B, N, D, key_range,
            min_dist.mutable_data_ptr<float>(),
            bucket_best_key.mutable_data_ptr<int64_t>());
    } else if (st == torch::headeronly::ScalarType::Half) {
        const at::Half* xptr = x_contig.const_data_ptr<at::Half>();
        init_best_and_mindist_kernel<at::Half><<<grid_points, block, shmem_init, stream>>>(
            xptr,
            bucket_id.const_data_ptr<int32_t>(),
            start_idx.const_data_ptr<int64_t>(),
            selected_mask.mutable_data_ptr<uint8_t>(),
            B, N, D, key_range,
            min_dist.mutable_data_ptr<float>(),
            bucket_best_key.mutable_data_ptr<int64_t>());
    } else {
        const at::BFloat16* xptr = x_contig.const_data_ptr<at::BFloat16>();
        init_best_and_mindist_kernel<at::BFloat16><<<grid_points, block, shmem_init, stream>>>(
            xptr,
            bucket_id.const_data_ptr<int32_t>(),
            start_idx.const_data_ptr<int64_t>(),
            selected_mask.mutable_data_ptr<uint8_t>(),
            B, N, D, key_range,
            min_dist.mutable_data_ptr<float>(),
            bucket_best_key.mutable_data_ptr<int64_t>());
    }
    CUDA_CHECK(cudaGetLastError());

    torch::stable::Tensor next_idx = torch::stable::new_empty(x_contig, {B64}, torch::headeronly::ScalarType::Long);

    // Iterations.
    for (int64_t s = 1; s < k; ++s) {
        if (s == k - 1) {
            reduce_bucket_best_kernel<<<B, 256, 0, stream>>>(
                bucket_best_key.const_data_ptr<int64_t>(),
                bucket_count.const_data_ptr<int32_t>(),
                selected_mask.mutable_data_ptr<uint8_t>(),
                B, N, key_range,
                next_idx.mutable_data_ptr<int64_t>(),
                out_indices.mutable_data_ptr<int64_t>(),
                (int)k, (int)s);
            CUDA_CHECK(cudaGetLastError());
            break;
        }

        reduce_and_active_mask_kernel<<<B, 256, 0, stream>>>(
            coords_f32.const_data_ptr<float>(),
            coords2_ptr,
            bucket_id.const_data_ptr<int32_t>(),
            bbox_min.const_data_ptr<float>(),
            bbox_max.const_data_ptr<float>(),
            bbox2_min_ptr,
            bbox2_max_ptr,
            bucket_best_key.mutable_data_ptr<int64_t>(),
            bucket_count.const_data_ptr<int32_t>(),
            selected_mask.mutable_data_ptr<uint8_t>(),
            B, N, p, key_range,
            next_idx.mutable_data_ptr<int64_t>(),
            out_indices.mutable_data_ptr<int64_t>(),
            (int)k, (int)s,
            active_mask.mutable_data_ptr<uint8_t>());
        CUDA_CHECK(cudaGetLastError());

        dim3 grid_leaves(key_range, B);
        size_t shmem_update = (size_t)D * ((st == torch::headeronly::ScalarType::Float) ? sizeof(float) : sizeof(uint16_t));

        if (st == torch::headeronly::ScalarType::Float) {
            const float* xptr = x_contig.const_data_ptr<float>();
            if (coords2_ptr != nullptr) {
                update_leaves_kernel<float, true><<<grid_leaves, block, shmem_update, stream>>>(
                    xptr,
                    coords_f32.const_data_ptr<float>(),
                    coords2_ptr,
                    p,
                    bucket_offsets.const_data_ptr<int32_t>(),
                    bucket_indices.const_data_ptr<int32_t>(),
                    active_mask.const_data_ptr<uint8_t>(),
                    selected_mask.mutable_data_ptr<uint8_t>(),
                    next_idx.const_data_ptr<int64_t>(),
                    B, N, D, key_range,
                    min_dist.mutable_data_ptr<float>(),
                    bucket_best_key.mutable_data_ptr<int64_t>());
            } else {
                update_leaves_kernel<float, false><<<grid_leaves, block, shmem_update, stream>>>(
                    xptr,
                    coords_f32.const_data_ptr<float>(),
                    nullptr,
                    p,
                    bucket_offsets.const_data_ptr<int32_t>(),
                    bucket_indices.const_data_ptr<int32_t>(),
                    active_mask.const_data_ptr<uint8_t>(),
                    selected_mask.mutable_data_ptr<uint8_t>(),
                    next_idx.const_data_ptr<int64_t>(),
                    B, N, D, key_range,
                    min_dist.mutable_data_ptr<float>(),
                    bucket_best_key.mutable_data_ptr<int64_t>());
            }
        } else if (st == torch::headeronly::ScalarType::Half) {
            const at::Half* xptr = x_contig.const_data_ptr<at::Half>();
            if (coords2_ptr != nullptr) {
                update_leaves_kernel<at::Half, true><<<grid_leaves, block, shmem_update, stream>>>(
                    xptr,
                    coords_f32.const_data_ptr<float>(),
                    coords2_ptr,
                    p,
                    bucket_offsets.const_data_ptr<int32_t>(),
                    bucket_indices.const_data_ptr<int32_t>(),
                    active_mask.const_data_ptr<uint8_t>(),
                    selected_mask.mutable_data_ptr<uint8_t>(),
                    next_idx.const_data_ptr<int64_t>(),
                    B, N, D, key_range,
                    min_dist.mutable_data_ptr<float>(),
                    bucket_best_key.mutable_data_ptr<int64_t>());
            } else {
                update_leaves_kernel<at::Half, false><<<grid_leaves, block, shmem_update, stream>>>(
                    xptr,
                    coords_f32.const_data_ptr<float>(),
                    nullptr,
                    p,
                    bucket_offsets.const_data_ptr<int32_t>(),
                    bucket_indices.const_data_ptr<int32_t>(),
                    active_mask.const_data_ptr<uint8_t>(),
                    selected_mask.mutable_data_ptr<uint8_t>(),
                    next_idx.const_data_ptr<int64_t>(),
                    B, N, D, key_range,
                    min_dist.mutable_data_ptr<float>(),
                    bucket_best_key.mutable_data_ptr<int64_t>());
            }
        } else {
            const at::BFloat16* xptr = x_contig.const_data_ptr<at::BFloat16>();
            if (coords2_ptr != nullptr) {
                update_leaves_kernel<at::BFloat16, true><<<grid_leaves, block, shmem_update, stream>>>(
                    xptr,
                    coords_f32.const_data_ptr<float>(),
                    coords2_ptr,
                    p,
                    bucket_offsets.const_data_ptr<int32_t>(),
                    bucket_indices.const_data_ptr<int32_t>(),
                    active_mask.const_data_ptr<uint8_t>(),
                    selected_mask.mutable_data_ptr<uint8_t>(),
                    next_idx.const_data_ptr<int64_t>(),
                    B, N, D, key_range,
                    min_dist.mutable_data_ptr<float>(),
                    bucket_best_key.mutable_data_ptr<int64_t>());
            } else {
                update_leaves_kernel<at::BFloat16, false><<<grid_leaves, block, shmem_update, stream>>>(
                    xptr,
                    coords_f32.const_data_ptr<float>(),
                    nullptr,
                    p,
                    bucket_offsets.const_data_ptr<int32_t>(),
                    bucket_indices.const_data_ptr<int32_t>(),
                    active_mask.const_data_ptr<uint8_t>(),
                    selected_mask.mutable_data_ptr<uint8_t>(),
                    next_idx.const_data_ptr<int64_t>(),
                    B, N, D, key_range,
                    min_dist.mutable_data_ptr<float>(),
                    bucket_best_key.mutable_data_ptr<int64_t>());
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }

    return out_indices;
}

// ----
// Baseline vanilla FPS (CUDA). We keep the algorithm identical to the prior
// baseline implementation, but adapt the interface to our stable-ABI wrapper.
// ----

template <unsigned int block_size>
__global__ void FarthestPointSamplingKernelBaselineStable(
    const float* __restrict__ points, // [B, N, D]
    const int64_t* __restrict__ start_idxs, // [B]
    const uint8_t* __restrict__ invalid_mask, // [B, N] (1=invalid) or nullptr
    int64_t N,
    int64_t D,
    int64_t k,
    int64_t* __restrict__ idxs, // [B, k]
    float* __restrict__ min_point_dist // [B, N]
) {
    typedef cub::BlockReduce<
        cub::KeyValuePair<int64_t, float>,
        block_size,
        cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>
        BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ int64_t selected_store;

    const int64_t b = (int64_t)blockIdx.x;
    const size_t tid = (size_t)threadIdx.x;

    int64_t selected = start_idxs[b];
    if (tid == 0) {
        idxs[b * k + 0] = selected;
    }

    for (int64_t kk = 1; kk < k; ++kk) {
        int64_t max_dist_idx = 0;
        float max_dist = -1.0f;

        for (int64_t p = (int64_t)tid; p < N; p += (int64_t)block_size) {
            if (invalid_mask && invalid_mask[b * N + p]) {
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
            idxs[b * k + kk] = selected;
            selected_store = selected;
        }
        __syncthreads();
        selected = selected_store;
    }
}

static torch::stable::Tensor sample_idx_cuda_baseline_impl(
    const torch::stable::Tensor& x,
    int64_t k,
    const torch::stable::Tensor& start_idx,
    const torch::stable::Tensor& invalid_mask
) {
    _check_cuda_tensor(x, "x");
    _check_cuda_tensor(start_idx, "start_idx");
    _check_cuda_tensor(invalid_mask, "invalid_mask");
    _check_same_device(x, start_idx, "x", "start_idx");
    _check_same_device(x, invalid_mask, "x", "invalid_mask");

    STD_TORCH_CHECK(x.dim() == 3, "x must be 3D [B,N,D]");
    STD_TORCH_CHECK(k >= 1, "k must be >= 1, got ", k);

    auto x_contig = torch::stable::contiguous(x);

    const int64_t B = x_contig.size(0);
    const int64_t N = x_contig.size(1);
    const int64_t D = x_contig.size(2);

    STD_TORCH_CHECK(k <= N, "k must be <= N. Got k=", k, " N=", N);

    STD_TORCH_CHECK(start_idx.scalar_type() == torch::headeronly::ScalarType::Long,
                    "start_idx must have dtype int64");
    STD_TORCH_CHECK(invalid_mask.scalar_type() == torch::headeronly::ScalarType::Byte,
                    "invalid_mask must have dtype uint8");

    STD_TORCH_CHECK(start_idx.dim() == 1 && start_idx.size(0) == B,
                    "start_idx must have shape [B]");
    STD_TORCH_CHECK(invalid_mask.dim() == 2 && invalid_mask.size(0) == B && invalid_mask.size(1) == N,
                    "invalid_mask must have shape [B,N]");

    CUDA_CHECK(cudaSetDevice(x.get_device_index()));
    cudaStream_t stream = _get_current_stream_or_throw(x.get_device_index());

    // Baseline operates on float32 points.
    auto x_f32 = torch::stable::contiguous(torch::stable::to(x_contig, torch::headeronly::ScalarType::Float));

    auto idxs = torch::stable::new_empty(x_contig, {B, k}, torch::headeronly::ScalarType::Long);
    torch::stable::fill_(idxs, -1.0);

    auto min_point_dist = torch::stable::new_empty(x_contig, {B, N}, torch::headeronly::ScalarType::Float);
    torch::stable::fill_(min_point_dist, 1e10);

    const int points_pow_2 = (N > 1) ? (int)std::log2((double)N) : 0;
    const int MAX_THREADS_PER_BLOCK = 1024;
    const int threads = (int)std::max<int64_t>(2, std::min<int64_t>((int64_t)(1 << points_pow_2), (int64_t)MAX_THREADS_PER_BLOCK));

    const float* points_ptr = x_f32.const_data_ptr<float>();
    const int64_t* start_ptr = start_idx.const_data_ptr<int64_t>();
    const uint8_t* invalid_ptr = invalid_mask.const_data_ptr<uint8_t>();

    const size_t blocks = (size_t)B;

    switch (threads) {
        case 1024:
            FarthestPointSamplingKernelBaselineStable<1024><<<blocks, threads, 0, stream>>>(
                points_ptr, start_ptr, invalid_ptr, N, D, k,
                idxs.mutable_data_ptr<int64_t>(), min_point_dist.mutable_data_ptr<float>());
            break;
        case 512:
            FarthestPointSamplingKernelBaselineStable<512><<<blocks, threads, 0, stream>>>(
                points_ptr, start_ptr, invalid_ptr, N, D, k,
                idxs.mutable_data_ptr<int64_t>(), min_point_dist.mutable_data_ptr<float>());
            break;
        case 256:
            FarthestPointSamplingKernelBaselineStable<256><<<blocks, threads, 0, stream>>>(
                points_ptr, start_ptr, invalid_ptr, N, D, k,
                idxs.mutable_data_ptr<int64_t>(), min_point_dist.mutable_data_ptr<float>());
            break;
        case 128:
            FarthestPointSamplingKernelBaselineStable<128><<<blocks, threads, 0, stream>>>(
                points_ptr, start_ptr, invalid_ptr, N, D, k,
                idxs.mutable_data_ptr<int64_t>(), min_point_dist.mutable_data_ptr<float>());
            break;
        case 64:
            FarthestPointSamplingKernelBaselineStable<64><<<blocks, threads, 0, stream>>>(
                points_ptr, start_ptr, invalid_ptr, N, D, k,
                idxs.mutable_data_ptr<int64_t>(), min_point_dist.mutable_data_ptr<float>());
            break;
        case 32:
            FarthestPointSamplingKernelBaselineStable<32><<<blocks, threads, 0, stream>>>(
                points_ptr, start_ptr, invalid_ptr, N, D, k,
                idxs.mutable_data_ptr<int64_t>(), min_point_dist.mutable_data_ptr<float>());
            break;
        case 16:
            FarthestPointSamplingKernelBaselineStable<16><<<blocks, threads, 0, stream>>>(
                points_ptr, start_ptr, invalid_ptr, N, D, k,
                idxs.mutable_data_ptr<int64_t>(), min_point_dist.mutable_data_ptr<float>());
            break;
        case 8:
            FarthestPointSamplingKernelBaselineStable<8><<<blocks, threads, 0, stream>>>(
                points_ptr, start_ptr, invalid_ptr, N, D, k,
                idxs.mutable_data_ptr<int64_t>(), min_point_dist.mutable_data_ptr<float>());
            break;
        case 4:
            FarthestPointSamplingKernelBaselineStable<4><<<blocks, threads, 0, stream>>>(
                points_ptr, start_ptr, invalid_ptr, N, D, k,
                idxs.mutable_data_ptr<int64_t>(), min_point_dist.mutable_data_ptr<float>());
            break;
        case 2:
            FarthestPointSamplingKernelBaselineStable<2><<<blocks, threads, 0, stream>>>(
                points_ptr, start_ptr, invalid_ptr, N, D, k,
                idxs.mutable_data_ptr<int64_t>(), min_point_dist.mutable_data_ptr<float>());
            break;
        default:
            // Should never happen because threads is clamped to a power-of-two in [2,1024].
            FarthestPointSamplingKernelBaselineStable<256><<<blocks, 256, 0, stream>>>(
                points_ptr, start_ptr, invalid_ptr, N, D, k,
                idxs.mutable_data_ptr<int64_t>(), min_point_dist.mutable_data_ptr<float>());
            break;
    }
    CUDA_CHECK(cudaGetLastError());

    return idxs;
}

} // anonymous namespace

STABLE_TORCH_LIBRARY_IMPL(torch_quickfps, CUDA, m) {
    m.impl("_sample_idx_impl", TORCH_BOX(&sample_idx_cuda_bucket_impl));
    m.impl("_sample_idx_baseline_impl", TORCH_BOX(&sample_idx_cuda_baseline_impl));
}
