#if __cplusplus < 201703L
#error "C++17 is required"
#endif

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/library.h>

#define STR_(x) #x
#define STR(x) STR_(x)

TORCH_LIBRARY(torch_quickfps, m) {
    // low_d controls the KD-tree bucketing space on CUDA.
    // If low_d is None, the CUDA path attempts to build the KD-tree in the
    // original feature space (C=D). For high-dimensional embeddings, users
    // should set low_d to a small value (e.g., 3 or 8) so bucketing/pruning
    // runs in a cheap projected space while distances are still computed in
    // full D.
    m.def("sample(Tensor self, int k, int? h=None, int? start_idx=None, Tensor? mask=None, int? low_d=None) -> (Tensor, Tensor)");
    m.def("sample_idx(Tensor self, int k, int? h=None, int? start_idx=None, Tensor? mask=None, int? low_d=None) -> Tensor");
    m.def("sample_baseline(Tensor self, int k, int? h=None, int? start_idx=None, Tensor? mask=None, int? low_d=None) -> (Tensor, Tensor)");
    m.def("sample_idx_baseline(Tensor self, int k, int? h=None, int? start_idx=None, Tensor? mask=None, int? low_d=None) -> Tensor");
}

PYBIND11_MODULE(_core, m) {
    m.attr("CPP_VERSION") = __cplusplus;
    m.attr("PYTORCH_VERSION") = STR(TORCH_VERSION_MAJOR) "." STR(
        TORCH_VERSION_MINOR) "." STR(TORCH_VERSION_PATCH);
    m.attr("PYBIND11_VERSION") = STR(PYBIND11_VERSION_MAJOR) "." STR(
        PYBIND11_VERSION_MINOR) "." STR(PYBIND11_VERSION_PATCH);
}
