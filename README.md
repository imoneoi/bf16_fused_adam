# BFloat16 Fused Optimizer

A mixed-precision optimizer to solve the [stale weights]() problem of bfloat16 training.

Compared to PyTorch FusedAdamW, this optimizer stores an extra 16-bit weights mantissa, acting as 16+16 optimizer, which is mathematically equivalent to storing a 32-bit master weight while more memory efficient.

PyTorch FusedAdamW:

 - param (bf16)
 - grad (bf16)
 - exp_avg (bf16)
 - exp_avg_sq (bf16)

16+16 Fused:

 - param (bf16, high 16 bits of master fp32 weights)
 - mantissa (int16, low 16 bits of master fp32 weights)
 - grad (bf16)
 - exp_avg (bf16)
 - exp_avg_sq (bf16)

```
Master weight: (1 bit sign) (8 bit exponent) (7 bit mantissa) (16 bit mantissa)
               [            param (16)                      ] [ mantissa (16) ]
```

References:

PyTorch AdamW:
 - https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/fused_adam_utils.cuh
 - https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/FusedAdamWKernel.cu

Gopher:
 - https://storage.googleapis.com/deepmind-media/research/language-research/Training%20Gopher.pdf