// vec_add_bind.cpp
#include <torch/extension.h>


void vecAdd(torch::Tensor A, torch::Tensor B, torch::Tensor C);
void fusedOp(torch::Tensor A, torch::Tensor B, torch::Tensor C);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vec_add", &vecAdd, "vector addition (CUDA)");
    m.def("fused_op", &fusedOp, "Fused operation: sinf(A*B) + log1p(A+B)");
}