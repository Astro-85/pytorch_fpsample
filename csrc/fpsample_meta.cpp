#include <torch/extension.h>
#include <torch/library.h>

using torch::Tensor;

std::tuple<Tensor, Tensor> sample_meta(const Tensor &x, int64_t k,
                                       torch::optional<int64_t> h,
                                       torch::optional<int64_t> start_idx,
                                       torch::optional<Tensor> mask,
                                       torch::optional<int64_t> low_d) {
    TORCH_CHECK(x.dim() >= 2,
                "x must have at least 2 dims, but got size: ", x.sizes());
    TORCH_CHECK(k >= 1, "k must be greater than or equal to 1, but got ", k);
    (void)h;
    (void)start_idx;
    (void)mask;
    (void)low_d;

    auto tmp_s1 = x.sizes().vec();
    tmp_s1[tmp_s1.size() - 2] = k;

    auto tmp_s2 = x.sizes().vec();
    tmp_s2.pop_back();
    tmp_s2[tmp_s2.size() - 1] = k;

    return std::make_tuple(
        torch::empty(tmp_s1, x.options()),
        torch::empty(tmp_s2, x.options().dtype(torch::kLong)));
}

Tensor sample_idx_meta(const Tensor &x, int64_t k,
                       torch::optional<int64_t> h,
                       torch::optional<int64_t> start_idx,
                       torch::optional<Tensor> mask,
                       torch::optional<int64_t> low_d) {
    TORCH_CHECK(x.dim() >= 2,
                "x must have at least 2 dims, but got size: ", x.sizes());
    TORCH_CHECK(k >= 1, "k must be greater than or equal to 1, but got ", k);
    (void)h;
    (void)start_idx;
    (void)mask;
    (void)low_d;

    auto tmp_s2 = x.sizes().vec();
    tmp_s2.pop_back();
    tmp_s2[tmp_s2.size() - 1] = k;
    return torch::empty(tmp_s2, x.options().dtype(torch::kLong));
}

TORCH_LIBRARY_IMPL(torch_quickfps, Meta, m) {
    m.impl("sample", &sample_meta);
    m.impl("sample_idx", &sample_idx_meta);
    m.impl("sample_baseline", &sample_meta);
    m.impl("sample_idx_baseline", &sample_idx_meta);
}
