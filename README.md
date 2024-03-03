# BFloat16 Fused Optimizer

A mixed-precision optimizer to solve the [stale weights](https://storage.googleapis.com/deepmind-media/research/language-research/Training%20Gopher.pdf) problem of bfloat16 training.

Compared to PyTorch FusedAdamW, this optimizer stores an extra 16-bit weights mantissa, acting as 16+16 optimizer, which is mathematically equivalent to storing a 32-bit master weight while being more memory efficient.

PyTorch FusedAdamW States:

 - param (bf16)
 - grad (bf16)
 - exp_avg (bf16)
 - exp_avg_sq (bf16)

16+16 Fused States:

 - param (bf16, high 16 bits of master fp32 weights)
 - mantissa (int16, low 16 bits of master fp32 weights)
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
               [             param 16           ] [mantissa 8] [truncated 8]   = 24bit
```

## References

16+16 optimizer:

 - https://arxiv.org/pdf/2309.12381.pdf

PyTorch AdamW:
 - https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/fused_adam_utils.cuh
 - https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/FusedAdamWKernel.cu

Gopher:
 - https://storage.googleapis.com/deepmind-media/research/language-research/Training%20Gopher.pdf
