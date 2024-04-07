from typing import Tuple
import torch

import math
import pytest

from .adamw import _bf16_fused_adamw


def _bf16_adamw_reference_impl(
    # FP32
    param: torch.Tensor,
    # BF16
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    # Constant
    step_size: float,
    wd_step_size: float,
    bias_correction2_sqrt: float,
    beta1: float,
    beta2: float,
    eps: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Cast to fp32.
    grad = grad.to(torch.float32)

    # Math
    # Reference implementation (PyTorch):
    # https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py
    param.mul_(1 - wd_step_size)

    exp_avg.lerp_(grad, 1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
    param.addcdiv_(exp_avg, denom, value=-step_size)

    return param, exp_avg, exp_avg_sq


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
    grad_std=0.001,
    atol=0.0007,
    steps=100
):
    torch.random.manual_seed(0)

    initial_param = torch.empty(params_shape, dtype=torch.bfloat16, device="cuda").normal_(std=init_std).to(torch.float32)

    # Reference
    ref_param = initial_param.clone()
    ref_exp_avg = torch.zeros_like(ref_param, dtype=torch.float32)
    ref_exp_avg_sq = torch.zeros_like(ref_param, dtype=torch.float32)
    ref_steps = 0

    # Test
    test_param = ref_param.to(torch.bfloat16)

    test_residual = torch.zeros_like(test_param)
    test_exp_avg = torch.zeros_like(test_param)
    test_exp_avg_sq = torch.zeros_like(test_param)
    test_steps = torch.zeros((), dtype=torch.float32, device="cuda")

    for _ in range(steps):
        grad = torch.empty(params_shape, dtype=torch.bfloat16, device="cuda").normal_(std=grad_std)

        # Reference
        ref_steps += 1
        ref_param, ref_exp_avg, ref_exp_avg_sq = _bf16_adamw_reference_impl(
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
            [test_residual],
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

    # Info
    print (f"[shape {params_shape} lr {lr} eps {eps}] Mean update magnitude: {torch.mean(torch.abs(initial_param - ref_exp_avg))}, atol: {atol}")

    # Check
    assert torch.allclose(test_param.to(torch.float32), ref_param, rtol=0, atol=atol)
    assert torch.allclose(test_exp_avg.to(torch.float32), ref_exp_avg, rtol=0, atol=atol)
    assert torch.allclose(test_exp_avg_sq.to(torch.float32), ref_exp_avg_sq, rtol=0, atol=atol)
    assert test_steps.item() == ref_steps
