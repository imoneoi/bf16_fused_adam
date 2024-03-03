import torch

import math
import pytest

from adamw import _bf16_fused_adamw


def _bf16_adamw_reference_impl(
    # FP32
    param,
    # BF16
    grad,
    exp_avg,
    exp_avg_sq,
    # Constant
    step_size,
    wd_step_size,
    bias_correction2_sqrt,
    beta1,
    beta2,
    eps
):
    # Cast to fp32.
    grad = grad.to(torch.float32)
    exp_avg = exp_avg.to(torch.float32)
    exp_avg_sq = exp_avg_sq.to(torch.float32)

    # Math
    param.sub_(wd_step_size * param)

    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad

    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt) + eps
    param.sub_(step_size * exp_avg / denom)

    return param, exp_avg.to(torch.bfloat16), exp_avg_sq.to(torch.bfloat16)


@pytest.mark.parametrize("params_shape", [(1, ), (4096, ), (4096, 14336)])
@pytest.mark.parametrize("lr", [1e-3, 1e-4, 5e-4])
@pytest.mark.parametrize("eps", [1e-5, 1e-8])
def test_bf16_adamw_backend(
    params_shape,
    lr,
    eps,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    init_std=0.02,
    grad_std=1e-3,
    steps=10
):
    # Reference
    ref_param = torch.empty(params_shape, dtype=torch.float32, device="cuda").normal_(std=init_std)
    ref_exp_avg = torch.zeros_like(ref_param, dtype=torch.bfloat16)
    ref_exp_avg_sq = torch.zeros_like(ref_param, dtype=torch.bfloat16)
    ref_steps = 0

    # Test
    test_param = (ref_param.view(dtype=torch.int32) >> 16).to(torch.int16).view(dtype=torch.bfloat16)
    test_mantissa = ref_param.view(dtype=torch.int32).to(torch.int16).view(dtype=torch.bfloat16)

    test_exp_avg = torch.zeros_like(test_param)
    test_exp_avg_sq = torch.zeros_like(test_param)
    test_steps = torch.zeros((), dtype=torch.float32, device="cuda")

    for _ in range(steps):
        grad = torch.empty(params_shape, dtype=torch.bfloat16, device="cuda").normal_(std=grad_std)

        # Reference
        ref_steps += 1
        _bf16_adamw_reference_impl(
            ref_param,
            grad,
            ref_exp_avg,
            ref_exp_avg_sq,
            step_size=lr / (1 - beta1 ** ref_steps),
            wd_step_size=lr * weight_decay,
            bias_correction2_sqrt=math.sqrt(1 - beta2 ** ref_steps),
            beta1=beta1,
            beta2=beta2,
            eps=eps
        )

        # Test
        _bf16_fused_adamw(
            [test_param],
            [test_mantissa],
            [grad],
            [test_exp_avg],
            [test_exp_avg_sq],
            [test_steps],
            beta1,
            beta2,
            lr,
            weight_decay,
            eps
        )

    # Check
    print ()
