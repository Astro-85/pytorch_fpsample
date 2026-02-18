// Workaround / support TU for the PyTorch Stable C++ API.
//
// Some stable-ABI Tensor methods are declared in torch/csrc/stable/tensor.h but
// require a single translation unit to include the header to ensure the
// definitions are emitted consistently (see PyTorch stable-ABI docs / known
// linker issues around tensor.h inclusion).
//
// Keep this file minimal and compile it exactly once.

#include <torch/csrc/stable/tensor.h>
