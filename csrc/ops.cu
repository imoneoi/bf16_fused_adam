#include "bf16_fused_adam.h"

#include <torch/extension.h>


namespace bf16_fused_adam {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bf16_fused_adamw_cuda_impl_", &bf16_fused_adamw_cuda_impl_, "BFloat16 Fused AdamW Implementation");
}

}  // namespace bf16_fused_adam
