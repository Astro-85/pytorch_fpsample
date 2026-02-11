
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAContext.h>
// NOTE: ATen/cuda/CUDAGuard.h was removed in newer PyTorch versions.
// Use the c10 header instead (works for both host and nvcc compilation).
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <type_traits>
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
// non-expansive (a contraction) in L2:
//   y = x @ R, with ||R||_2 <= ||R||_F = 1.
//
// Concretely, each projection vector has entries in {+1,-1} and is scaled by
// 1/sqrt(D*p). This guarantees ||y(x)-y(z)|| <= ||x-z||, which keeps the
// pruning bound in active_mask_kernel safe.
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
    constexpr int MAX_P = 32;
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

// Assign bucket ids by traversing a complete binary KD-tree.
//
// The tree is stored as arrays over internal nodes (size = key_range-1).
// For a node index n: left=2n+1, right=2n+2.
// After `height` levels, the node index points into the leaf region.
// The corresponding leaf id is node - (key_range-1).
__global__ void assign_bucket_id_kdtree_kernel(
    const float* __restrict__ coords,      // [B,N,p] float32
    const int32_t* __restrict__ split_dim, // [B,internal_nodes]
    const float* __restrict__ split_val,   // [B,internal_nodes]
    const uint8_t* __restrict__ invalid_mask, // [B,N] uint8 (1=invalid)
    int B, int N, int p,
    int height,
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

    const int internal_nodes = key_range - 1;
    int node = 0;
    const float* ci = coords + ((int64_t)b * N + i) * p;

    for (int d = 0; d < height; ++d) {
        int32_t dim = split_dim[b * internal_nodes + node];
        float thr = split_val[b * internal_nodes + node];
        // Guard against degenerate dim indices.
        dim = (dim < 0) ? 0 : (dim >= p ? (p - 1) : dim);
        float v = ci[dim];
        node = (v <= thr) ? (2 * node + 1) : (2 * node + 2);
        if (node >= (2 * internal_nodes + 1)) break;
    }

    int leaf = node - internal_nodes;
    if (leaf < 0) leaf = 0;
    if (leaf >= key_range) leaf = key_range - 1;
    bucket_id[(int64_t)b * N + i] = (int32_t)leaf;
}

__global__ void compute_bucket_id_kernel(
    const float* __restrict__ coords, // [B,N,p] float32
    const float* __restrict__ minv, // [B,p]
    const float* __restrict__ maxv, // [B,p]
    int B, int N, int p, int q,
    int32_t* __restrict__ bucket_id // [B,N]
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || i >= N) return;

    const float* ci = coords + ((int64_t)b * N + i) * p;
    int32_t key = 0;

    if (p == 1) {
        float mn = minv[b * p + 0];
        float mx = maxv[b * p + 0];
        float range = mx - mn;
        range = (range > 1e-12f) ? range : 1.0f;
        float v = (ci[0] - mn) / range;
        v = fminf(fmaxf(v, 0.0f), 0.999999f);
        int32_t ix = (int32_t)floorf(v * q);
        ix = min(max(ix, 0), q - 1);
        key = ix;
    } else if (p == 2) {
        float mn0 = minv[b * p + 0], mx0 = maxv[b * p + 0];
        float mn1 = minv[b * p + 1], mx1 = maxv[b * p + 1];
        float r0 = mx0 - mn0; r0 = (r0 > 1e-12f) ? r0 : 1.0f;
        float r1 = mx1 - mn1; r1 = (r1 > 1e-12f) ? r1 : 1.0f;
        float v0 = (ci[0] - mn0) / r0;
        float v1 = (ci[1] - mn1) / r1;
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
        float v0 = (ci[0] - mn0) / r0;
        float v1 = (ci[1] - mn1) / r1;
        float v2 = (ci[2] - mn2) / r2;
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

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            if (sh_best_k[t + stride] > sh_best_k[t]) sh_best_k[t] = sh_best_k[t + stride];
        }
        __syncthreads();
    }

    if (t == 0) {
        unsigned long long best_k = sh_best_k[0];
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

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            if (sh_best_k[t + stride] > sh_best_k[t]) sh_best_k[t] = sh_best_k[t + stride];
        }
        __syncthreads();
    }

    __shared__ int64_t sh_chosen;
    if (t == 0) {
        unsigned long long best_k = sh_best_k[0];
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

__global__ void active_mask_kernel(
    const float* __restrict__ coords1, // [B,N,p] float32
    const float* __restrict__ coords2, // [B,N,p] float32 (optional; may be nullptr)
    const int32_t* __restrict__ bucket_id, // [B,N]
    const int64_t* __restrict__ ref_idx, // [B]
    const float* __restrict__ bbox1_min, // [B,key_range,p]
    const float* __restrict__ bbox1_max, // [B,key_range,p]
    const float* __restrict__ bbox2_min, // [B,key_range,p] (optional; may be nullptr)
    const float* __restrict__ bbox2_max, // [B,key_range,p] (optional; may be nullptr)
    int64_t* __restrict__ bucket_best_key, // [B,key_range]
    const int32_t* __restrict__ bucket_count, // [B,key_range]
    int B, int N, int p, int key_range,
    uint8_t* __restrict__ active_mask // [B,key_range]
) {
    int b = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || j >= key_range) return;

    int32_t cnt = bucket_count[b * key_range + j];

    int64_t ridx = ref_idx[b];
    ridx = (ridx < 0) ? 0 : ((ridx >= N) ? (N - 1) : ridx);
    int32_t ref_bucket = bucket_id[(int64_t)b * N + ridx];

    if (cnt <= 0) {
        active_mask[b * (int64_t)key_range + j] = 0;
        return;
    }

    unsigned long long key = (unsigned long long)bucket_best_key[b * (int64_t)key_range + j];
    float lastmax = unpack_best_dist(key);

    // Fast reject: if lastmax==0, nothing in this bucket can improve.
    if (!(lastmax > 0.0f)) {
        uint8_t active = (ref_bucket == j) ? 1 : 0;
        active_mask[b * (int64_t)key_range + j] = active;
        if (active) bucket_best_key[b * (int64_t)key_range + j] = 0;
        return;
    }

    const float* cref1 = coords1 + ((int64_t)b * N + ridx) * p;

    const int64_t leaf_base = ((int64_t)b * key_range + j) * p;
    const float* b1min = bbox1_min + leaf_base;
    const float* b1max = bbox1_max + leaf_base;

    // Compute squared distance from the reference point to the axis-aligned
    // bounding box of the bucket in the bucketing space (projection 1).
    float bound = 0.0f;
    for (int d = 0; d < p; ++d) {
        float v = cref1[d];
        float dd = 0.0f;
        float lo = b1min[d], hi = b1max[d];
        if (v > hi) dd = v - hi;
        else if (v < lo) dd = lo - v;
        bound += dd * dd;
    }

    // Optional second projection: take max(lower_bound_1, lower_bound_2)
    // for a tighter (still safe) bound on the full-D distance.
    if (coords2 != nullptr && bbox2_min != nullptr && bbox2_max != nullptr) {
        const float* cref2 = coords2 + ((int64_t)b * N + ridx) * p;
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

    // Safe pruning.
    uint8_t active = (bound < lastmax) ? 1 : 0;

    // Always update the bucket that contains the reference index.
    if (ref_bucket == j) active = 1;

    active_mask[b * (int64_t)key_range + j] = active;

    // Fuse reset: active buckets will be recomputed from scratch in the update kernel.
    if (active) {
        bucket_best_key[b * (int64_t)key_range + j] = 0;
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
            for (int dd = 0; dd < 64; ++dd) {
                if (dd >= p) break;
                float dv = ci1[dd] - cref1[dd];
                lb += dv * dv;
            }
            if constexpr (USE_COORD2) {
                float lb2 = 0.0f;
                const float* ci2 = coords2 + ((int64_t)b * N + i) * p;
                #pragma unroll
                for (int dd = 0; dd < 64; ++dd) {
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



// -------------------------
// Active leaf compaction + persistent per-leaf update
// -------------------------

// Compact active leaves into a flat list of global bucket indices: g = b * key_range + leaf.
// Single-block kernel: total elements (B*key_range) is small in typical settings.
__global__ void compact_active_pairs_kernel(
    const uint8_t* __restrict__ active_mask, // [B,key_range]
    int B, int key_range,
    int32_t* __restrict__ active_pairs, // [B*key_range]
    int32_t* __restrict__ active_total  // [1]
) {
    if (blockIdx.x != 0) return;
    if (threadIdx.x == 0) active_total[0] = 0;
    __syncthreads();

    int total = B * key_range;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        if (active_mask[idx]) {
            int pos = atomicAdd(&active_total[0], 1);
            active_pairs[pos] = idx;
        }
    }
}

template <typename scalar_t>
__global__ void update_active_pairs_persistent_kernel(
    const scalar_t* __restrict__ x, // [B,N,D]
    const int32_t* __restrict__ bucket_offsets, // [B,key_range+1]
    const int32_t* __restrict__ bucket_indices, // [B,N]
    const uint8_t* __restrict__ selected_mask, // [B,N]
    const int64_t* __restrict__ ref_idx, // [B]
    int B, int N, int D, int key_range,
    const int32_t* __restrict__ active_pairs, // [*]
    const int32_t* __restrict__ active_total, // [1]
    int32_t* __restrict__ work_counter, // [1] (initialized to 0)
    float* __restrict__ min_dist, // [B,N]
    int64_t* __restrict__ bucket_best_key // [B,key_range]
) {
    __shared__ int task;
    while (true) {
        if (threadIdx.x == 0) {
            task = atomicAdd(work_counter, 1);
        }
        __syncthreads();
        if (task >= active_total[0]) return;

        int g = active_pairs[task];
        int b = g / key_range;
        int leaf = g - b * key_range;
        if (b < 0 || b >= B || leaf < 0 || leaf >= key_range) {
            __syncthreads();
            continue;
        }

        int64_t ridx = ref_idx[b];
        ridx = (ridx < 0) ? 0 : ((ridx >= N) ? (N - 1) : ridx);

        const scalar_t* xb = x + ((int64_t)b * N * D);
        const scalar_t* xref = xb + (int64_t)ridx * D;

        int64_t base_o = (int64_t)b * (key_range + 1);
        int32_t start = bucket_offsets[base_o + leaf];
        int32_t end   = bucket_offsets[base_o + (leaf + 1)];
        if (start < 0) start = 0;
        if (end > N) end = N;

        unsigned long long local_best = 0ull;

        for (int32_t t = start + (int32_t)threadIdx.x; t < end; t += (int32_t)blockDim.x) {
            int32_t i = bucket_indices[(int64_t)b * N + t];
            if ((int)i < 0 || (int)i >= N) continue;
            if (selected_mask && selected_mask[(int64_t)b * N + i]) continue;

            float old = min_dist[(int64_t)b * N + i];
            float dist = 0.0f;
            const scalar_t* xi = xb + (int64_t)i * D;

            constexpr int CHUNK = 32;
            int d = 0;
            for (; d + CHUNK <= D; d += CHUNK) {
                #pragma unroll
                for (int dd = 0; dd < CHUNK; ++dd) {
                    float dv = to_float(xi[d + dd]) - to_float(xref[d + dd]);
                    dist += dv * dv;
                }
                if (dist >= old) { dist = old; break; }
            }
            if (dist < old) {
                for (; d < D; ++d) {
                    float dv = to_float(xi[d]) - to_float(xref[d]);
                    dist += dv * dv;
                    if (dist >= old) { dist = old; break; }
                }
            }

            float nd = dist;
            if (nd < old) {
                min_dist[(int64_t)b * N + i] = nd;
            } else {
                nd = old;
            }

            unsigned long long k = pack_best_key(nd, i);
            if (k > local_best) local_best = k;
        }

        unsigned long long v = warp_reduce_max_u64(local_best);
        __shared__ unsigned long long warp_best[32];
        int lane = threadIdx.x & 31;
        int wid  = threadIdx.x >> 5;
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
        __syncthreads();
    }
}



// -------------------------


// -------------------------
// CPU-side KD-tree split builder (per batch)
// -------------------------
//
// We build a complete binary KD-tree of fixed `height` in the bucketing space
// (dimension p). This produces `internal_nodes = 2^height - 1` splits.
//
// Split rule (matches CPU KDLineTree behavior):
//   * choose the dimension with the widest extent (max-min)
//   * split at the mean along that dimension
//   * partition indices by (v <= thr) vs (v > thr)
//
// Invalid points (invalid_mask[i]==1) are excluded from tree construction.
//
static void build_kdtree_splits_batch(
    const float* coords, // [N,p] contiguous
    int N, int p, int height,
    const uint8_t* invalid_mask, // [N] or nullptr, 1=invalid
    int32_t* split_dim, // [internal_nodes]
    float* split_val    // [internal_nodes]
) {
    const int key_range = 1 << height;
    const int internal_nodes = key_range - 1;

    for (int i = 0; i < internal_nodes; ++i) {
        split_dim[i] = 0;
        split_val[i] = 0.0f;
    }

    std::vector<int32_t> perm;
    perm.reserve((size_t)N);
    if (invalid_mask) {
        for (int i = 0; i < N; ++i) {
            if (!invalid_mask[i]) perm.push_back((int32_t)i);
        }
    } else {
        for (int i = 0; i < N; ++i) perm.push_back((int32_t)i);
    }

    const int MAX_P = 64; // enforced by caller
    struct Builder {
        const float* coords;
        int p;
        int height;
        int internal_nodes;
        std::vector<int32_t>& perm;
        int32_t* split_dim;
        float* split_val;

        void build(int node, int depth, int l, int r) {
            if (depth >= height) return;
            if (node < 0 || node >= internal_nodes) return;

            const int cnt = r - l;
            if (cnt <= 0) {
                build(2 * node + 1, depth + 1, l, l);
                build(2 * node + 2, depth + 1, l, l);
                return;
            }

            float mn[MAX_P];
            float mx[MAX_P];
            for (int d = 0; d < p; ++d) {
                mn[d] = std::numeric_limits<float>::infinity();
                mx[d] = -std::numeric_limits<float>::infinity();
            }

            // bbox
            for (int t = l; t < r; ++t) {
                int32_t idx = perm[(size_t)t];
                const float* ci = coords + (int64_t)idx * p;
                for (int d = 0; d < p; ++d) {
                    float v = ci[d];
                    mn[d] = (v < mn[d]) ? v : mn[d];
                    mx[d] = (v > mx[d]) ? v : mx[d];
                }
            }

            // widest dim
            int best_dim = 0;
            float best_range = mx[0] - mn[0];
            for (int d = 1; d < p; ++d) {
                float range = mx[d] - mn[d];
                if (range > best_range) { best_range = range; best_dim = d; }
            }

            // mean split
            double sum = 0.0;
            for (int t = l; t < r; ++t) {
                int32_t idx = perm[(size_t)t];
                sum += (double)coords[(int64_t)idx * p + best_dim];
            }
            float thr = (float)(sum / (double)cnt);

            split_dim[node] = (int32_t)best_dim;
            split_val[node] = thr;

            // partition
            int i = l;
            int j = r;
            while (i < j) {
                int32_t idx = perm[(size_t)i];
                float v = coords[(int64_t)idx * p + best_dim];
                if (v <= thr) {
                    ++i;
                } else {
                    --j;
                    std::swap(perm[(size_t)i], perm[(size_t)j]);
                }
            }
            int mid = i;

            build(2 * node + 1, depth + 1, l, mid);
            build(2 * node + 2, depth + 1, mid, r);
        }
    };

    Builder builder{coords, p, height, internal_nodes, perm, split_dim, split_val};
    builder.build(0, 0, 0, (int)perm.size());
}

// Host-side entrypoints
// -------------------------

static torch::Tensor sample_cuda_indices_impl(
    const torch::Tensor &x, int64_t k, torch::optional<int64_t> h,
    torch::optional<int64_t> start_idx,
    torch::optional<torch::Tensor> mask,
    torch::optional<int64_t> low_d) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor, but found on ", x.device());
    c10::cuda::CUDAGuard device_guard(x.device());
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims, but got size: ", x.sizes());
    TORCH_CHECK(k >= 1, "k must be >= 1, but got ", k);

    auto [old_size, x_reshaped_raw] = bnorm_reshape(x);
    auto x_reshaped = x_reshaped_raw.contiguous(); // keep original dtype

    TORCH_CHECK(x_reshaped.scalar_type() == torch::kFloat32 ||
                x_reshaped.scalar_type() == torch::kFloat16 ||
                x_reshaped.scalar_type() == torch::kBFloat16,
                "x must have dtype float32/float16/bfloat16 on CUDA, but got ", x_reshaped.scalar_type());

    const auto B = (int)x_reshaped.size(0);
    const auto N = (int)x_reshaped.size(1);
    const auto D = (int)x_reshaped.size(2);

    TORCH_CHECK(k <= N, "k must be <= N. Got k=", k, " N=", N);

    int64_t height = h.value_or(5);
    height = std::max<int64_t>(1, height);
    TORCH_CHECK(height <= 12,
                "h (KD-tree height) must be <= 12 for CUDA (max 4096 leaves), got ", height);
    const int key_range = 1 << (int)height;
    const int internal_nodes = key_range - 1;

    // KD-tree bucketing dimension.
    // If low_d is provided, we bucket in a rademacher-projected space of size p.
    // Otherwise, we bucket in the full D-dimensional space (only supported for modest D).
    int64_t low_d_v = low_d.value_or(-1);
    int p = (low_d_v > 0) ? (int)std::min<int64_t>((int64_t)D, low_d_v) : D;
    TORCH_CHECK(p >= 1, "p must be >= 1, got ", p);
    // NOTE: bbox/count tensors scale with p, so we cap this for practical performance.
    TORCH_CHECK(p <= 64,
                "For CUDA KD-tree bucketing, bucketing dimension p must be <= 64. "
                "Got p=", p, " (D=", D, "). Please set low_d<=64 (e.g., 3 or 8) for high-D embeddings.");

    auto opts_i32 = x_reshaped.options().dtype(torch::kInt32);
    auto opts_i64 = x_reshaped.options().dtype(torch::kInt64);
    auto opts_f32 = x_reshaped.options().dtype(torch::kFloat32);
    auto opts_u8  = x_reshaped.options().dtype(torch::kUInt8);

    torch::Tensor mask_b;
    torch::Tensor selected_mask;
    if (mask.has_value() && mask.value().defined()) {
        auto m = mask.value();
        TORCH_CHECK(m.is_cuda(), "mask must be a CUDA tensor when x is CUDA");
        TORCH_CHECK(m.device() == x.device(), "mask must be on the same device as x");
        TORCH_CHECK(m.scalar_type() == torch::kBool || m.scalar_type() == torch::kUInt8,
                    "mask must have dtype bool or uint8, but got ", m.scalar_type());
        TORCH_CHECK(m.numel() == (int64_t)B * (int64_t)N,
                    "mask must have shape (*, N) matching x's batch/point dims. Expected numel=",
                    (int64_t)B * (int64_t)N, " but got numel=", m.numel());
        mask_b = m.to(torch::kBool).contiguous().view({B, N});

        auto counts = mask_b.sum(1); // [B]
        auto min_valid = std::get<0>(counts.min(0)).item<int64_t>();
        TORCH_CHECK(min_valid >= k,
                    "mask has fewer than k valid points in at least one batch. min_valid=",
                    min_valid, " k=", k);

        selected_mask = (~mask_b).to(torch::kUInt8).contiguous(); // 1=not eligible
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
            start_idx_t = mask_b.to(torch::kInt64).argmax(1);
        } else {
            start_idx_t = torch::randint(0, N, {B}, x_reshaped.options().dtype(torch::kInt64));
        }
    }

    const int threads = 256;
    dim3 block(threads);
    dim3 grid_points((N + threads - 1) / threads, B);
    dim3 grid_bucket((key_range + threads - 1) / threads, B);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.device().index()).stream();

    // Build coordinates used for KD-tree bucketing + pruning.
    // If p==D, this is just x (cast to float32). If p<D, use a cheap rademacher
    // projection to p dims (safe for pruning because it's non-expansive).
    torch::Tensor coords_f32;
    torch::Tensor coords2_f32;

    if (p == D) {
        // Full-D coordinates already provide a tight bound; no need for a second projection.
        coords_f32 = x_reshaped.to(torch::kFloat32).contiguous();
    } else {
        coords_f32  = torch::empty({B, N, p}, opts_f32);
        coords2_f32 = torch::empty({B, N, p}, opts_f32);

        // Two independent Rademacher projections. Each projected distance is a LOWER BOUND on the
        // full-D distance (because the projection is scaled to be non-expansive), so using
        // max(dist1, dist2) yields a tighter (still safe) pruning bound.
        const uint32_t seed1 = 0x6d2b79f5u ^ (uint32_t)D * 0x9e3779b9u ^ (uint32_t)p * 0x85ebca6bu;
        const uint32_t seed2 = seed1 ^ 0x68bc21ebu;

        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                        x_reshaped.scalar_type(), "torch_fpsample_proj2", [&] {
            const scalar_t* xptr = (const scalar_t*)x_reshaped.data_ptr<scalar_t>();
            project_rademacher_kernel<scalar_t><<<grid_points, block, 0, stream>>>(
                xptr, B, N, D, p, seed1, coords_f32.data_ptr<float>());
            project_rademacher_kernel<scalar_t><<<grid_points, block, 0, stream>>>(
                xptr, B, N, D, p, seed2, coords2_f32.data_ptr<float>());
        });
        CUDA_CHECK(cudaGetLastError());
    }

    torch::Tensor bucket_id = torch::full({B, N}, -1, opts_i32);

    torch::Tensor bbox_min = torch::full({B, key_range, p}, INFINITY, opts_f32);
    torch::Tensor bbox_max = torch::full({B, key_range, p}, -INFINITY, opts_f32);
    torch::Tensor bbox2_min;
    torch::Tensor bbox2_max;
    if (coords2_f32.defined()) {
        bbox2_min = torch::full({B, key_range, p}, INFINITY, opts_f32);
        bbox2_max = torch::full({B, key_range, p}, -INFINITY, opts_f32);
    }

    torch::Tensor bucket_count = torch::zeros({B, key_range}, opts_i32);

    // CSR representation of bucket membership (built once per call).
    torch::Tensor bucket_offsets = torch::empty({B, key_range + 1}, opts_i32);
    torch::Tensor bucket_indices = torch::empty({B, N}, opts_i32);
    torch::Tensor bucket_cursor  = torch::zeros({B, key_range}, opts_i32);


    torch::Tensor bucket_best_key = torch::zeros({B, key_range}, opts_i64);

    torch::Tensor active_mask = torch::zeros({B, key_range}, opts_u8);
    torch::Tensor min_dist = torch::empty({B, N}, opts_f32);

    torch::Tensor out_indices = torch::empty({B, (int)k}, opts_i64);
    out_indices.select(1, 0).copy_(start_idx_t);

    // ---- KD-tree bucketing (adaptive, GPU build, complete tree of height `h`) ----
    // Build a complete binary KD-tree level-by-level in the bucketing space.
    // This avoids GPU->CPU sync/copies (a major source of overhead in v8) while
    // keeping the FPS update path fully parallel on GPU.
    const uint8_t* invalid_ptr = selected_mask.defined() ? selected_mask.data_ptr<uint8_t>() : nullptr;

    torch::Tensor node_id = torch::zeros({B, N}, opts_i32);
    torch::Tensor node_min = torch::empty({B, key_range, p}, opts_f32);
    torch::Tensor node_max = torch::empty({B, key_range, p}, opts_f32);
    torch::Tensor node_sum = torch::empty({B, key_range, p}, opts_f32);
    torch::Tensor node_count = torch::empty({B, key_range}, opts_i32);
    torch::Tensor split_dim_lvl = torch::empty({B, key_range}, opts_i32);
    torch::Tensor split_val_lvl = torch::empty({B, key_range}, opts_f32);

    for (int depth = 0; depth < (int)height; ++depth) {
        int nodes = 1 << depth;
        int total = nodes * p;
        dim3 grid_stats((total + threads - 1) / threads, B);
        kdtree_reset_stats_kernel<<<grid_stats, block, 0, stream>>>(
            B, key_range, nodes, p,
            node_min.data_ptr<float>(),
            node_max.data_ptr<float>(),
            node_sum.data_ptr<float>(),
            node_count.data_ptr<int32_t>());
        CUDA_CHECK(cudaGetLastError());

        kdtree_accum_stats_kernel<<<grid_points, block, 0, stream>>>(
            coords_f32.data_ptr<float>(),
            node_id.data_ptr<int32_t>(),
            invalid_ptr,
            B, N, p,
            nodes,
            key_range,
            node_min.data_ptr<float>(),
            node_max.data_ptr<float>(),
            node_sum.data_ptr<float>(),
            node_count.data_ptr<int32_t>());
        CUDA_CHECK(cudaGetLastError());

        dim3 grid_nodes(((nodes + threads - 1) / threads), B);
        kdtree_compute_splits_kernel<<<grid_nodes, block, 0, stream>>>(
            B,
            nodes,
            key_range,
            p,
            node_min.data_ptr<float>(),
            node_max.data_ptr<float>(),
            node_sum.data_ptr<float>(),
            node_count.data_ptr<int32_t>(),
            split_dim_lvl.data_ptr<int32_t>(),
            split_val_lvl.data_ptr<float>());
        CUDA_CHECK(cudaGetLastError());

        kdtree_update_node_id_kernel<<<grid_points, block, 0, stream>>>(
            coords_f32.data_ptr<float>(),
            split_dim_lvl.data_ptr<int32_t>(),
            split_val_lvl.data_ptr<float>(),
            invalid_ptr,
            B, N, p,
            nodes,
            key_range,
            node_id.data_ptr<int32_t>());
        CUDA_CHECK(cudaGetLastError());
    }

    kdtree_finalize_bucket_id_kernel<<<grid_points, block, 0, stream>>>(
        node_id.data_ptr<int32_t>(),
        invalid_ptr,
        B, N,
        key_range,
        bucket_id.data_ptr<int32_t>());
    CUDA_CHECK(cudaGetLastError());

        const float* coords2_ptr = coords2_f32.defined() ? coords2_f32.data_ptr<float>() : nullptr;
    float* bbox2_min_ptr = bbox2_min.defined() ? bbox2_min.data_ptr<float>() : nullptr;
    float* bbox2_max_ptr = bbox2_max.defined() ? bbox2_max.data_ptr<float>() : nullptr;

    bbox_counts_kernel<<<grid_points, block, 0, stream>>>(
        coords_f32.data_ptr<float>(),
        coords2_ptr,
        bucket_id.data_ptr<int32_t>(),
        B, N, p, key_range,
        bbox_min.data_ptr<float>(),
        bbox_max.data_ptr<float>(),
        bbox2_min_ptr,
        bbox2_max_ptr,
        bucket_count.data_ptr<int32_t>());
    CUDA_CHECK(cudaGetLastError());

    // Build CSR: bucket_offsets + bucket_indices.
    // Offsets: exclusive scan over bucket_count (tiny, done by one thread per batch).
    build_bucket_offsets_kernel<<<B, 1, 0, stream>>>(
        bucket_count.data_ptr<int32_t>(),
        B, key_range,
        bucket_offsets.data_ptr<int32_t>());
    CUDA_CHECK(cudaGetLastError());

    // Indices: counting-sort by bucket using atomic cursors.
    fill_bucket_indices_kernel<<<grid_points, block, 0, stream>>>(
        bucket_id.data_ptr<int32_t>(),
        bucket_offsets.data_ptr<int32_t>(),
        bucket_cursor.data_ptr<int32_t>(),
        B, N, key_range,
        bucket_indices.data_ptr<int32_t>());
    CUDA_CHECK(cudaGetLastError());


    size_t shmem = (size_t)D * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                   x_reshaped.scalar_type(), "torch_fpsample_cuda", [&] {
        const scalar_t* xptr = (const scalar_t*)x_reshaped.data_ptr<scalar_t>();

        init_best_and_mindist_kernel<scalar_t><<<grid_points, block, shmem, stream>>>(
            xptr,
            bucket_id.data_ptr<int32_t>(),
            start_idx_t.data_ptr<int64_t>(),
            selected_mask.data_ptr<uint8_t>(),
            B, N, D, key_range,
            min_dist.data_ptr<float>(),
            bucket_best_key.data_ptr<int64_t>());
        CUDA_CHECK(cudaGetLastError());

        torch::Tensor next_idx = torch::empty({B}, opts_i64);

        for (int64_t s = 1; s < k; s++) {
            if (s == k - 1) {
                reduce_bucket_best_kernel<<<B, 256, 0, stream>>>(
                    bucket_best_key.data_ptr<int64_t>(),
                    bucket_count.data_ptr<int32_t>(),
                    selected_mask.data_ptr<uint8_t>(),
                    B, N, key_range,
                    next_idx.data_ptr<int64_t>(),
                    out_indices.data_ptr<int64_t>(), (int)k, (int)s);
                CUDA_CHECK(cudaGetLastError());
                break;
            }

            reduce_and_active_mask_kernel<<<B, 256, 0, stream>>>(
                coords_f32.data_ptr<float>(),
                coords2_ptr,
                bucket_id.data_ptr<int32_t>(),
                bbox_min.data_ptr<float>(),
                bbox_max.data_ptr<float>(),
                bbox2_min_ptr,
                bbox2_max_ptr,
                bucket_best_key.data_ptr<int64_t>(),
                bucket_count.data_ptr<int32_t>(),
                selected_mask.data_ptr<uint8_t>(),
                B, N, p, key_range,
                next_idx.data_ptr<int64_t>(),
                out_indices.data_ptr<int64_t>(), (int)k, (int)s,
                active_mask.data_ptr<uint8_t>());
            CUDA_CHECK(cudaGetLastError());

            // Fix A: drop per-iteration active-leaf compaction + persistent work queue.
            // For typical settings (key_range<=256), launching one block per leaf is cheap.
            // Inactive leaves return immediately.
            dim3 grid_leaves(key_range, B);
            size_t shmem_update = (size_t)D * sizeof(scalar_t);

            if (coords2_ptr != nullptr) {
                update_leaves_kernel<scalar_t, true><<<grid_leaves, block, shmem_update, stream>>>(
                    xptr,
                    coords_f32.data_ptr<float>(),
                    coords2_ptr,
                    p,
                    bucket_offsets.data_ptr<int32_t>(),
                    bucket_indices.data_ptr<int32_t>(),
                    active_mask.data_ptr<uint8_t>(),
                    selected_mask.data_ptr<uint8_t>(),
                    next_idx.data_ptr<int64_t>(),
                    B, N, D, key_range,
                    min_dist.data_ptr<float>(),
                    bucket_best_key.data_ptr<int64_t>());
            } else {
                update_leaves_kernel<scalar_t, false><<<grid_leaves, block, shmem_update, stream>>>(
                    xptr,
                    coords_f32.data_ptr<float>(),
                    nullptr,
                    p,
                    bucket_offsets.data_ptr<int32_t>(),
                    bucket_indices.data_ptr<int32_t>(),
                    active_mask.data_ptr<uint8_t>(),
                    selected_mask.data_ptr<uint8_t>(),
                    next_idx.data_ptr<int64_t>(),
                    B, N, D, key_range,
                    min_dist.data_ptr<float>(),
                    bucket_best_key.data_ptr<int64_t>());
            }
            CUDA_CHECK(cudaGetLastError());

        }
    });

    auto ret_indices_sizes = old_size.vec();
    ret_indices_sizes.pop_back();
    ret_indices_sizes[ret_indices_sizes.size() - 1] = k;

    return out_indices.view(ret_indices_sizes).to(torch::kLong);
}

} // anonymous namespace

torch::Tensor sample_idx_cuda(const torch::Tensor &x, int64_t k,
                              torch::optional<int64_t> h,
                              torch::optional<int64_t> start_idx,
                              torch::optional<torch::Tensor> mask,
                              torch::optional<int64_t> low_d) {
    return sample_cuda_indices_impl(x, k, h, start_idx, mask, low_d);
}

std::tuple<torch::Tensor, torch::Tensor> sample_cuda(
    const torch::Tensor &x, int64_t k, torch::optional<int64_t> h,
    torch::optional<int64_t> start_idx,
    torch::optional<torch::Tensor> mask,
    torch::optional<int64_t> low_d) {

    auto idx = sample_cuda_indices_impl(x, k, h, start_idx, mask, low_d);

    // Gather sampled points from the original (possibly non-float32) view.
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
