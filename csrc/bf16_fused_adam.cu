#include "bf16_fused_adam.h"

#include <cuda_bf16.h>

#include <ATen/core/Tensor.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <ATen/native/cuda/Pow.cuh>
#include <utility>


namespace bf16_fused_adam {

using bfloat16_t = __nv_bfloat16;
using at::native::kILP;

constexpr int kArgsDepth = 5;

constexpr uint8_t kParamIdx = 0;
constexpr uint8_t kResidualIdx = 1;
constexpr uint8_t kGradIdx = 2;
constexpr uint8_t kExpAvgIdx = 3;
constexpr uint8_t kExpAvgSqIdx = 4;

__device__ __forceinline__ bfloat16_t lerp(const bfloat16_t &v0, const bfloat16_t &v1, const bfloat16_t &t) {
    // NOTE(one): Identical to PyTorch when t < 0.5
    // https://github.com/pytorch/pytorch/blob/b7f25226929e70187a9f36c393665abad0b25190/aten/src/ATen/native/Lerp.h#L21
    return __hfma(t, v1, __hfma(-t, v0, v0));
}

__device__ __forceinline__ void adamw_math(
    bfloat16_t r_args[kArgsDepth][kILP],
    const bfloat16_t &step_size,
    const bfloat16_t &wd_step_size,
    const bfloat16_t &mbeta1,
    const bfloat16_t &mbeta2,
    const bfloat16_t &eps,
    const bfloat16_t &bias_correction2_sqrt)
{
#pragma unroll
    for (int ii = 0; ii < kILP; ii++)
    {
        bfloat16_t param = r_args[kParamIdx][ii];
        bfloat16_t residual = r_args[kResidualIdx][ii];

        bfloat16_t grad = r_args[kGradIdx][ii];

        bfloat16_t exp_avg = r_args[kExpAvgIdx][ii];
        bfloat16_t exp_avg_sq = r_args[kExpAvgSqIdx][ii];

        // Adam Momentums
        exp_avg = lerp(exp_avg, grad, mbeta1);
        exp_avg_sq = lerp(exp_avg_sq, grad * grad, mbeta2);

        // Update Size
        bfloat16_t denom = (sqrt(exp_avg_sq) / bias_correction2_sqrt) + eps;
        bfloat16_t update = step_size * exp_avg / denom + wd_step_size * param;

        // Kahan Summation
        bfloat16_t kahan_increment = update - residual;
        bfloat16_t new_param = param + kahan_increment;
        bfloat16_t new_residual = (new_param - param) - kahan_increment;

        // Store results.
        r_args[kParamIdx][ii] = new_param;
        r_args[kResidualIdx][ii] = new_residual;

        r_args[kExpAvgIdx][ii] = exp_avg;
        r_args[kExpAvgSqIdx][ii] = exp_avg_sq;
    }
}

struct FusedAdamMathFunctor {
  __device__ __forceinline__ void operator()(
      int chunk_size,
      at::native::FusedOptimizerTensorListMetadata<kArgsDepth>& tl,
      const double& lr,
      const double& beta1,
      const double& beta2,
      const double& weight_decay,
      const double& eps) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];

    const auto [step_size, wd_step_size, bias_correction2_sqrt, mbeta1, mbeta2, meps] = [&]() -> std::tuple<bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t, bfloat16_t> {
      auto* step_count = reinterpret_cast<const float*>(tl.state_steps_addresses[tensor_loc]);
      const auto bias_correction1 = 1 - at::native::pow_(beta1, *step_count);
      const auto bias_correction2 = 1 - at::native::pow_(beta2, *step_count);
      const auto bias_correction2_sqrt = std::sqrt(bias_correction2);

      return {-lr / bias_correction1, -lr * weight_decay, bias_correction2_sqrt, 1 - beta1, 1 - beta2, eps};
    }();

    bfloat16_t* args[kArgsDepth];
    bfloat16_t r_args[kArgsDepth][kILP];
    const auto n = tl.numel_for_tensor[tensor_loc] - chunk_idx * chunk_size;

    const bool all_aligned{
        at::native::init_args<kArgsDepth>(args, tl, chunk_idx, chunk_size, tensor_loc)};
    if ((n % kILP == 0) && (chunk_size % kILP == 0) && all_aligned) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
#pragma unroll
        for (int i = 0; i < kArgsDepth; i++) {
          at::native::load_store(r_args[i], args[i], 0, i_start);
        }
        adamw_math(
            r_args,
            step_size,
            wd_step_size,
            mbeta1,
            mbeta2,
            meps,
            bias_correction2_sqrt);
#pragma unroll
        for (int i = 0; i < kArgsDepth; i++) {
          if (i != kGradIdx) {
            at::native::load_store(args[i], r_args[i], i_start, 0);
          }
        }
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
        at::native::load_args<kArgsDepth>(r_args, args, i_start, chunk_size, n);
        adamw_math(
            r_args,
            step_size,
            wd_step_size,
            mbeta1,
            mbeta2,
            meps,
            bias_correction2_sqrt);
#pragma unroll
        for (int i = 0; i < kArgsDepth; i++) {
          if (i != kGradIdx) {
            at::native::store_args(args[i], r_args[i], i_start, chunk_size, n);
          }
        }
      }
    }
  }
};

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
    const double eps) {
  std::vector<std::vector<at::Tensor>> tensor_lists{params, mantissas, grads, exp_avgs, exp_avg_sqs};

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16,
      params[0].scalar_type(),
      "bf16_fused_adamw_kernel_cuda",
      [&]() {
        at::native::multi_tensor_apply_for_fused_optimizer<5>(
            tensor_lists,
            state_steps,
            FusedAdamMathFunctor(),
            lr,
            beta1,
            beta2,
            weight_decay,
            eps);
      });
}

} // namespace bf16_fused_adam