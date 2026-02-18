// ABI-stable operator registration + CPython module stub.
//
// This translation unit intentionally contains:
//   1) STABLE_TORCH_LIBRARY schemas for internal kernels.
//   2) A minimal CPython module init (py_limited_api) so the .so can be
//      packaged and loaded via torch.ops.load_library.
//
// The public torch_quickfps operators (sample/sample_idx/...) are defined in
// Python as CompositeExplicitAutograd in torch_quickfps/_stable_register.py.

#include <Python.h>

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY(torch_quickfps, m) {
    // Internal (compiled) kernels.
    m.def(
        "_sample_idx_impl(Tensor x, int k, int h, Tensor start_idx, Tensor invalid_mask, int low_d) -> Tensor");
    m.def(
        "_sample_idx_baseline_impl(Tensor x, int k, Tensor start_idx, Tensor invalid_mask) -> Tensor");
}

// ---- CPython stub -------------------------------------------------------

static struct PyModuleDef _core_module = {
    PyModuleDef_HEAD_INIT,
    "_core",
    nullptr,
    -1,
    nullptr,
};

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&_core_module);
}
