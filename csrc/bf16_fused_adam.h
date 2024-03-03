#include <torch/extension.h>


namespace bf16_fused_adam {

void bf16_fused_adamw_cuda_impl_(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> mantissas,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    std::vector<at::Tensor> state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps);

}  // namespace bf16_fused_adam
