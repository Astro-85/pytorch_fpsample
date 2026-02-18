
#pragma once
#include <cstddef>
#include <cstdint>

void bucket_fps_kdline(const float *raw_data, size_t n_points, size_t dim,
                       size_t n_samples, size_t start_idx, size_t height,
                       int64_t *sampled_point_indices);


/*
// csrc/cpu/bucket_fps/wrapper.h
#pragma once
#include <cstddef>
#include <cstdint>

void kdline_sample_varlen(const float* raw_data,
                          std::size_t n_points,
                          std::size_t dim,
                          std::size_t n_samples,
                          std::size_t start_idx,
                          std::size_t height,
                          int64_t* sampled_point_indices);

void kdline_sample_varlen_stable(const float* raw_data,
                                 std::size_t n_points,
                                 std::size_t dim,
                                 std::size_t n_samples,
                                 std::size_t start_idx,
                                 std::size_t height,
                                 int64_t* sampled_point_indices);
*/
