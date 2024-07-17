from low_rank_extraction_fuse_compression_quantization import low_rank_addition_fuse_compression_quantization
from low_rank_extraction_fuse_decompression_dequantization import low_rank_addition_fuse_decompression_dequantization
import torch
from copy import deepcopy

def naive_quantization_dequantization(l, r, x, s, quantize_bit=8, outlier=5.):
    x = x - l @ r
    # step 1: extract the outlier
    outlier_mask = torch.abs(x) > outlier
    o = x * outlier_mask
    x = x - o
    # step 2: compress the rest part
    # s = s.unsqueeze(-2)
    q = torch.clamp(torch.round(x / s), min = -2 ** (quantize_bit-1), max = 2 ** (quantize_bit-1) - 1).to(torch.int32)
    
    x = q.to(torch.bfloat16) * s + o + l @ r
    return x

if __name__ == '__main__':
    M, R = 1024, 16
    B = 1
    l = torch.randn(M, R, device='cuda', dtype=torch.bfloat16) 
    r = torch.randn(R, M, device='cuda', dtype=torch.bfloat16) 
    x = torch.randn(B, M, M, device='cuda', dtype=torch.bfloat16) 
    x_copy = deepcopy(x)
    s = torch.zeros(B, M, device='cuda', dtype=torch.bfloat16) + 1
    s[:, :20] = 1.2
    s[:, 20:] = 1.5

    o, q = low_rank_addition_fuse_compression_quantization(l, r, x, s)
    x_decode = low_rank_addition_fuse_decompression_dequantization(l, r, q, o, s)
    x_baseline = naive_quantization_dequantization(l, r, x_copy, s)
    
    print(x_decode - x_baseline)
    x_diff = x_decode - x_baseline
    # print the sparse ratio
    print(torch.sum(x_diff == 0).item() / x_diff.numel())
    # print(torch.sum(x_decode - x_baseline))