#pragma once

#include <torch/extension.h>
#include <torch/library.h>

using torch::Tensor;

// CUDA implementation of torch_fpsample::sample.
// Signature must match the operator schema.
std::tuple<Tensor, Tensor> sample_cuda(const Tensor &x, int64_t k,
                                       torch::optional<int64_t> h,
                                       torch::optional<int64_t> start_idx,
                                       torch::optional<Tensor> mask);
