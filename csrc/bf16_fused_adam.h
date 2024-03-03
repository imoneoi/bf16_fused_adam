#include <torch/extension.h>


namespace bf16_fused_adam {

void bf16_fused_adamw_cuda_impl_(
    at::TensorList params,
    at::TensorList mantissas,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps);

}  // namespace bf16_fused_adam
