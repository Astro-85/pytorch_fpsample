#include <torch/extension.h>
#include <torch/library.h>

using torch::Tensor;
using FuncType = std::tuple<Tensor, Tensor>(const Tensor &, int64_t,
                                            torch::optional<int64_t>,
                                            torch::optional<int64_t>,
                                            torch::optional<Tensor>,
                                            torch::optional<int64_t>);
using FuncTypeBaseline = std::tuple<Tensor, Tensor>(const Tensor &, int64_t,
                                                    torch::optional<int64_t>,
                                                    torch::optional<int64_t>,
                                                    torch::optional<Tensor>,
                                                    torch::optional<int64_t>);

////////////////////
//                //
//    Autograd    //
//                //
////////////////////
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;
class FPSampleFunction : public torch::autograd::Function<FPSampleFunction> {
  public:
    static variable_list forward(AutogradContext *ctx, const Tensor &x,
                                 int64_t k, torch::optional<int64_t> h,
                                 torch::optional<int64_t> start_idx,
                                 torch::optional<Tensor> mask,
                                 torch::optional<int64_t> low_d) {
        torch::AutoDispatchBelowADInplaceOrView guard;
        static auto op = torch::Dispatcher::singleton()
                     .findSchemaOrThrow("torch_quickfps::sample", "")
                             .typed<FuncType>();
        auto results = op.call(x, k, h, start_idx, mask, low_d);
        auto ret_tensor = std::get<0>(results);
        auto ret_indices = std::get<1>(results);
        ctx->save_for_backward({ret_indices});
        ctx->saved_data["x_sizes"] = x.sizes();
        return {ret_tensor, ret_indices};
    }

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_outputs) {
        auto saved_tensors = ctx->get_saved_variables();
        auto ret_indices = saved_tensors[0];
        auto x_sizes = ctx->saved_data["x_sizes"].toIntVector();
        auto grad_output = grad_outputs[0];

        auto tmp_repeat_sizes = x_sizes;
        std::fill(tmp_repeat_sizes.begin(), tmp_repeat_sizes.end() - 1, 1);

        auto grad_input = torch::scatter(
            torch::zeros(x_sizes, grad_output.options()), -2,
            ret_indices.unsqueeze(-1).repeat(tmp_repeat_sizes), grad_output);

        return {grad_input, Variable(), Variable(), Variable(), Variable(), Variable()};
    }
};

class FPSampleBaselineFunction : public torch::autograd::Function<FPSampleBaselineFunction> {
  public:
    static variable_list forward(AutogradContext *ctx, const Tensor &x,
                                 int64_t k, torch::optional<int64_t> h,
                                 torch::optional<int64_t> start_idx,
                                 torch::optional<Tensor> mask,
                                 torch::optional<int64_t> low_d) {
        torch::AutoDispatchBelowADInplaceOrView guard;
        static auto op = torch::Dispatcher::singleton()
                     .findSchemaOrThrow("torch_quickfps::sample_baseline", "")
                             .typed<FuncTypeBaseline>();
        auto results = op.call(x, k, h, start_idx, mask, low_d);
        auto ret_tensor = std::get<0>(results);
        auto ret_indices = std::get<1>(results);
        ctx->save_for_backward({ret_indices});
        ctx->saved_data["x_sizes"] = x.sizes();
        return {ret_tensor, ret_indices};
    }

    static variable_list backward(AutogradContext *ctx,
                                  variable_list grad_outputs) {
        auto saved_tensors = ctx->get_saved_variables();
        auto ret_indices = saved_tensors[0];
        auto x_sizes = ctx->saved_data["x_sizes"].toIntVector();
        auto grad_output = grad_outputs[0];

        auto tmp_repeat_sizes = x_sizes;
        std::fill(tmp_repeat_sizes.begin(), tmp_repeat_sizes.end() - 1, 1);

        auto grad_input = torch::scatter(
            torch::zeros(x_sizes, grad_output.options()), -2,
            ret_indices.unsqueeze(-1).repeat(tmp_repeat_sizes), grad_output);

        return {grad_input, Variable(), Variable(), Variable(), Variable(), Variable()};
    }
};

std::tuple<Tensor, Tensor> sample_autograd(const Tensor &x, int64_t k,
                                           torch::optional<int64_t> h,
                                           torch::optional<int64_t> start_idx,
                                           torch::optional<Tensor> mask,
                                           torch::optional<int64_t> low_d) {
    auto results = FPSampleFunction::apply(x, k, h, start_idx, mask, low_d);
    return std::make_tuple(results[0], results[1]);
}

std::tuple<Tensor, Tensor> sample_baseline_autograd(const Tensor &x, int64_t k,
                                                    torch::optional<int64_t> h,
                                                    torch::optional<int64_t> start_idx,
                                                    torch::optional<Tensor> mask,
                                                    torch::optional<int64_t> low_d) {
    auto results = FPSampleBaselineFunction::apply(x, k, h, start_idx, mask, low_d);
    return std::make_tuple(results[0], results[1]);
}



using FuncIdxType = Tensor(const Tensor &, int64_t,
                           torch::optional<int64_t>,
                           torch::optional<int64_t>,
                           torch::optional<Tensor>,
                           torch::optional<int64_t>);
using FuncIdxBaselineType = Tensor(const Tensor &, int64_t,
                                   torch::optional<int64_t>,
                                   torch::optional<int64_t>,
                                   torch::optional<Tensor>,
                                   torch::optional<int64_t>);

Tensor sample_idx_autograd(const Tensor &x, int64_t k,
                           torch::optional<int64_t> h,
                           torch::optional<int64_t> start_idx,
                           torch::optional<Tensor> mask,
                           torch::optional<int64_t> low_d) {
    // Indices are non-differentiable; we simply route to the underlying kernel below AD.
    torch::AutoDispatchBelowADInplaceOrView guard;
    static auto op = torch::Dispatcher::singleton()
                         .findSchemaOrThrow("torch_quickfps::sample_idx", "")
                         .typed<FuncIdxType>();
    return op.call(x, k, h, start_idx, mask, low_d);
}

Tensor sample_idx_baseline_autograd(const Tensor &x, int64_t k,
                                    torch::optional<int64_t> h,
                                    torch::optional<int64_t> start_idx,
                                    torch::optional<Tensor> mask,
                                    torch::optional<int64_t> low_d) {
    torch::AutoDispatchBelowADInplaceOrView guard;
    static auto op = torch::Dispatcher::singleton()
                         .findSchemaOrThrow("torch_quickfps::sample_idx_baseline", "")
                         .typed<FuncIdxBaselineType>();
    return op.call(x, k, h, start_idx, mask, low_d);
}
TORCH_LIBRARY_IMPL(torch_quickfps, Autograd, m) {
    m.impl("sample", &sample_autograd);
    m.impl("sample_idx", &sample_idx_autograd);
    m.impl("sample_baseline", &sample_baseline_autograd);
    m.impl("sample_idx_baseline", &sample_idx_baseline_autograd);
}


