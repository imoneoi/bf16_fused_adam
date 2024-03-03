#include "bf16_fused_adam.h"

#include <ATen/core/Tensor.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <ATen/native/cuda/Pow.cuh>
#include <utility>


namespace bf16_fused_adam {

constexpr int kArgsDepth = 5;

constexpr uint8_t kParamIdx = 0;
constexpr uint8_t kGradIdx = 1;
constexpr uint8_t kExpAvgIdx = 2;
constexpr uint8_t kExpAvgSqIdx = 3;
constexpr uint8_t kMantissaIdx = 4;

template <typename T>
__device__ __forceinline__ T lerp(const T v0, const T v1, const T t) {
    return fma(t, v1, fma(-t, v0, v0));
}

__device__ __forceinline__ float concat_float(const at::BFloat16 value, const at::BFloat16 mantissa) {
    return reinterpret_cast<float>(
        (static_cast<uint32_t>(reinterpret_cast<uint16_t>(value)) << 16) | 
         static_cast<uint32_t>(reinterpret_cast<uint16_t>(mantissa))
    );
}

__device__ __forceinline__ void split_float(const float f, at::BFloat16 &value, at::BFloat16 &mantissa) {
    value = reinterpret_cast<at::BFloat16>(static_cast<uint16_t>(reinterpret_cast<uint32_t>(f) >> 16));
    mantissa = reinterpret_cast<at::BFloat16>(static_cast<uint16_t>(reinterpret_cast<uint32_t>(f)));
}

__device__ __forceinline__ void adamw_math(
    at::BFloat16 r_args[kArgsDepth][kILP],
    const float &step_size,
    const float &wd_step_size,
    const float &beta1,
    const float &beta2,
    const float &weight_decay,
    const float &eps,
    const float &bias_correction2_sqrt)
{
#pragma unroll
    for (int ii = 0; ii < kILP; ii++)
    {
        // Load values.
        const float grad = static_cast<float>(r_args[kGradIdx][ii]);

        float param = concat_float(r_args[kParamIdx][ii], r_args[kMantissaIdx][ii]);

        float exp_avg = static_cast<float>(r_args[kExpAvgIdx][ii]);
        float exp_avg_sq = static_cast<float>(r_args[kExpAvgSqIdx][ii]);

        param -= wd_step_size * param;

        exp_avg = lerp(grad, exp_avg, beta1);
        exp_avg_sq = lerp(grad * grad, exp_avg_sq, beta2);

        const float denom = (std::sqrt(exp_avg_sq) / bias_correction2_sqrt) + eps;
        param -= step_size * exp_avg / denom;

        // Store results.
        split_float(param, r_args[kParamIdx][ii], r_args[kMantissaIdx][ii]);
        r_args[kExpAvgIdx][ii] = exp_avg;
        r_args[kExpAvgSqIdx][ii] = exp_avg_sq;
    }
}

struct FusedAdamMathFunctor {
  __device__ __forceinline__ void operator()(
      int chunk_size,
      FusedOptimizerTensorListMetadata<kArgsDepth>& tl,
      const double& lr,
      const double& beta1,
      const double& beta2,
      const double& weight_decay,
      const double& eps) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];

    const auto [step_size, wd_step_size, bias_correction2_sqrt] =
        [&]() -> std::tuple<double, double> {
      auto* step_count = reinterpret_cast<const float*>(tl.state_steps_addresses[tensor_loc]);
      const auto bias_correction1 = 1 - at::native::pow_(beta1, *step_count);
      const auto bias_correction2 = 1 - at::native::pow_(beta2, *step_count);
      const auto bias_correction2_sqrt = std::sqrt(bias_correction2);
      return {lr * bias_correction1, lr * weight_decay, bias_correction2_sqrt};
    }();

    at::BFloat16* args[kArgsDepth];
    at::BFloat16 r_args[kArgsDepth][kILP];
    const auto n = tl.numel_for_tensor[tensor_loc] - chunk_idx * chunk_size;

    const bool all_aligned{
        init_args<kArgsDepth>(args, tl, chunk_idx, chunk_size, tensor_loc)};
    if ((n % kILP == 0) && (chunk_size % kILP == 0) && all_aligned) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
#pragma unroll
        for (int i = 0; i < kArgsDepth; i++) {
          load_store(r_args[i], args[i], 0, i_start);
        }
        adam_math(
            r_args,
            step_size,
            wd_step_size,
            beta1,
            beta2,
            weight_decay,
            eps,
            bias_correction2_sqrt);
#pragma unroll
        for (int i = 0; i < kArgsDepth; i++) {
          if (i != kGradIdx) {
            load_store(args[i], r_args[i], i_start, 0);
          }
        }
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
        load_args<kArgsDepth>(r_args, args, i_start, chunk_size, n);
        adam_math(
            r_args,
            step_size,
            wd_step_size,
            beta1,
            beta2,
            weight_decay,
            eps,
            bias_correction2_sqrt);
#pragma unroll
        for (int i = 0; i < kArgsDepth; i++) {
          if (i != kGradIdx) {
            store_args(args[i], r_args[i], i_start, chunk_size, n);
          }
        }
      }
    }
  }
};

void bf16_fused_adamw_cuda_impl_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList exp_avgs,
    at::TensorList exp_avg_sqs,
    at::TensorList mantissas,
    at::TensorList state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params.vec(), grads.vec(), exp_avgs.vec(), exp_avg_sqs.vec(), mantissas.vec()};

  AT_DISPATCH_FLOATING_TYPES_AND(
      kBFloat16,
      params[0].scalar_type(),
      "bf16_fused_adamw_kernel_cuda",
      [&]() {
        multi_tensor_apply_for_fused_optimizer<5>(
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