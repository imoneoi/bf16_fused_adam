# BFloat16 Fused Optimizer

A mixed-precision optimizer to solve the [stale weights](https://storage.googleapis.com/deepmind-media/research/language-research/Training%20Gopher.pdf) problem of bfloat16 training.

When training models using `bfloat16` optimizer, updates might often be cancelled if it's small compared to weight in magnitude, leading to the stale weights problem, which significantly hurt performance. 

Utilizing the fact that the round-towards-zero (RTZ) result of a `float32` to `bfloat16` is the high 16 bits, this optimizer stores an extra 16-bit weights mantissa, acting as 16+16 optimizer, which is mathematically equivalent to storing an extra 32-bit master weight, solving the stale weights problem while only costs 25% more memory.

## Usage

Drop-in replacement of `torch.optim.AdamW`. All parameters need to be in `bfloat16`.
 
 - Doesn't support `foreach`, `fused` argument, as the optimizer is already fused
 - Doesn't support `amsgrad`, `maximize`, `capturable`, `differentiable` argument yet

```bash
pip install bf16_fused_adam
```

```python
from bf16_fused_adam import BF16FusedAdamW

# All supported arguments are listed below
optim = BF16FusedAdamW(model.parameters(),
    lr=1e-3,
    weight_decay=0.1,
    betas=(0.9, 0.95),
    eps=1e-5,
)
```

## Details

AdamW Reference States (PyTorch FusedAdamW):

 - param (bf16)
 - grad (bf16)
 - exp_avg (bf16)
 - exp_avg_sq (bf16)

16+16 Optimizer States (BF16FusedAdamW):

 - param (bf16, high 16 bits of master fp32 weights)
 - mantissa (uint16, low 16 bits of master fp32 weights)
 - grad (bf16)
 - exp_avg (bf16)
 - exp_avg_sq (bf16)

```
Master weight: (sign 1) (exponent 8) (mantissa 7) (mantissa 16)   = 32bit
               [             param 16           ] [mantissa 16]   = 32bit
```

## TODO

 - [ ] Stochastic rounding (trading precision for memory)
 - [ ] 16+8 optimizer (saving more memory)

 ```
Master weight: (sign 1) (exponent 8) (mantissa 7) (mantissa 8) (mantissa 8)   = 32bit
               [             param 16           ] [mantissa 8] [dropped 8]    = 24bit
```

## Consistency Tests

We tested the consistency against reference AdamW implementation. To run tests, clone this repository, run pytest:

```bash
pip install -e .
pytest
```

### Passed

 - [x] H100
 - [x] A100
 - [ ] RTX 4090 [TBD]
 - [ ] RTX 3090 [TBD]

## References

16+16 optimizer:

 - https://arxiv.org/pdf/2309.12381.pdf

PyTorch AdamW:
 - https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/fused_adam_utils.cuh
 - https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/FusedAdamWKernel.cu

Gopher:
 - https://storage.googleapis.com/deepmind-media/research/language-research/Training%20Gopher.pdf
