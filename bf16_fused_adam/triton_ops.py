# TODO(one): Deprecate these kernels after PyTorch supports unsigned dtypes
# https://github.com/pytorch/pytorch/issues/58734

import torch
import triton
import triton.language as tl


# Concat kernel
@triton.jit
def bit_concat_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr
               ):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.uint16, bitcast=True)
    y = tl.load(y_ptr + offsets, mask=mask).to(tl.uint16, bitcast=True)

    output = ((x.to(tl.uint32) << 16) | y.to(tl.uint32)).to(tl.float32, bitcast=True)

    tl.store(output_ptr + offsets, output, mask=mask)


def bit_concat(x: torch.Tensor, y: torch.Tensor):
    assert x.is_cuda and y.is_cuda
    assert x.dtype is torch.bfloat16
    assert y.dtype is torch.bfloat16

    output = torch.empty_like(x, dtype=torch.float32)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    bit_concat_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# Split kernel
@triton.jit
def bit_split_kernel(x_ptr,
               output_hi_ptr,
               output_lo_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr
               ):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.uint32, bitcast=True)

    output_hi = (x >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    output_lo = x.to(tl.uint16).to(tl.bfloat16, bitcast=True)

    tl.store(output_hi_ptr + offsets, output_hi, mask=mask)
    tl.store(output_lo_ptr + offsets, output_lo, mask=mask)


def bit_split(x: torch.Tensor):
    assert x.is_cuda
    assert x.dtype is torch.float32

    output_hi = torch.empty_like(x, dtype=torch.bfloat16)
    output_lo = torch.empty_like(x, dtype=torch.bfloat16)
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    bit_split_kernel[grid](x, output_hi, output_lo, n_elements, BLOCK_SIZE=1024)
    return output_hi, output_lo
