import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8, }, num_stages=1, 
                      num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def low_rank_addition_fuse_decompression_dequantization_kernel(
    # Pointers to matrices
    l_ptr, r_ptr, x_ptr, x_temp_ptr, o_ptr, q_ptr, s_ptr,
    # Matrix dimensions
    B, M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_lm` is how much to increase `l_ptr`
    # by to get the element one row down (A has M rows).
    stride_lm, stride_lk,
    stride_rk, stride_rn,
    stride_xb, stride_xm, stride_xn,
    stride_x_tempb, stride_x_tempm, stride_x_tempn,
    stride_ob, stride_om, stride_on,
    stride_qb, stride_qm, stride_qn,
    stride_sb, stride_sn, # scaling factor has no stride in the m dimension
    # compress params
    quantize_bit: tl.constexpr, elem_per_position: tl.constexpr, 
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, 
):
    # basic pointers
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_lm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    l_ptrs = l_ptr + (offs_lm[:, None] * stride_lm + offs_k[None, :] * stride_lk)
    r_ptrs = r_ptr + (offs_k[:, None] * stride_rk + offs_rn[None, :] * stride_rn)

    offs_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_ptrs = x_ptr + stride_xm * offs_xm[:, None] + stride_xn * offs_xn[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    o_ptrs = o_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :] + offs_b * stride_ob
    o_mask = (offs_b < B) & (offs_xm[:, None] < M) & (offs_xn[None, :] < N)

    # quantization operators
    offs_qm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_qn = pid_n * (BLOCK_SIZE_N // elem_per_position) + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    q_ptrs = q_ptr + stride_qm * offs_qm[:, None] + stride_qn * offs_qn[None, :] + offs_b * stride_qb
    q_mask = (offs_b < B) & (offs_qm[:, None] < M) & (offs_qn[None, :] < N // elem_per_position)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    offs_x_temp_m = offs_xm
    offs_x_temp_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    x_temp_ptrs = x_temp_ptr + stride_x_tempm * offs_x_temp_m[:, None] + stride_x_tempn * offs_x_temp_n[None, :] + offs_b * stride_x_tempb

    mask = (1 << quantize_bit) - 1

    # extract the quantized values
    for i in range(elem_per_position):
        x_temp_ptrs_new = x_temp_ptrs + i * (BLOCK_SIZE_N // elem_per_position)
        element_fake_int = tl.math.uint2float_rn((q & mask).to(tl.uint32))
        tl.store(x_temp_ptrs_new, element_fake_int)
        q = (q >> quantize_bit).to(tl.uint8)
        
    # dequantize
    offs_sn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    s_ptrs = s_ptr + stride_sn * offs_sn[None, :] + offs_b * stride_sb
    s_mask = (offs_b < B) & (offs_sn[None, :] < N)
    s = tl.load(s_ptrs, mask=s_mask, other=1.0)
    
    offs_x_temp_m = offs_xm
    offs_x_temp_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_temp_ptrs = x_temp_ptr + stride_xm * offs_x_temp_m[:, None] + stride_xn * offs_x_temp_n[None, :] + offs_b * stride_xb
    
    x = tl.load(x_temp_ptrs, mask=x_mask, other=0.0)
    x = x - 2 ** (quantize_bit - 1)
    x = x.to(tl.bfloat16)
    x = x * s
    
    accumulator = tl.load(o_ptrs, mask=o_mask, other=0.0) # load the outliers
    accumulator += x
    accumulator = accumulator.to(tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(l_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(r_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b, allow_tf32=True)
        # Advance the ptrs to the next K block.
        l_ptrs += BLOCK_SIZE_K * stride_lk
        r_ptrs += BLOCK_SIZE_K * stride_rk
        
    y = accumulator.to(tl.bfloat16)
    tl.store(x_ptrs, y, mask=x_mask)
    
    
def low_rank_addition_fuse_decompression_dequantization(l, r, q, o, s, quantize_bit=8, outlier=5.):
    assert l.shape[1] == r.shape[0], "Incompatible dimensions"
    assert l.is_contiguous(), "Matrix A must be contiguous"
    assert r.is_contiguous(), "Matrix B must be contiguous"
    M, K = l.shape
    K, N = r.shape
    B, _, _ = q.shape
    
    if K < 16:
        l = torch.cat([l, torch.zeros((M, 16 - K), device=l.device, dtype=l.dtype)], dim=1).contiguous()
        r = torch.cat([r, torch.zeros((16 - K, N), device=r.device, dtype=r.dtype)], dim=0).contiguous()
        K = 16
    
    # 1D launch kernel where each block gets its own program.
    elem_per_position = 8 // quantize_bit
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    x = torch.empty((B, M, N), device=l.device, dtype=torch.bfloat16)
    x_temp = torch.empty((B, M, N), device=l.device, dtype=torch.uint8)
    
    low_rank_addition_fuse_decompression_dequantization_kernel[grid](
        l, r, x, x_temp, o, q, s,
        B, M, N, K,
        l.stride(0), l.stride(1),
        r.stride(0), r.stride(1),
        x.stride(0), x.stride(1), x.stride(2),
        x_temp.stride(0), x_temp.stride(1), x_temp.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(0), s.stride(1),
        quantize_bit, elem_per_position,
        BLOCK_SIZE_K=K
    )
    del x_temp
    return x


def torch_low_rank_addition_fuse_decompression_dequantization(l, r, q, o, s, quantize_bit=8):
    element_num = 8 // quantize_bit
    x = torch.empty((q.shape[0], q.shape[1], q.shape[2] * element_num), device=q.device, dtype=torch.int8)
    mask = (1 << quantize_bit) - 1
    for i in range(element_num):
        selected_index = torch.arange(q.shape[2] * i, q.shape[2] * (i + 1))
        x[:, :, selected_index] = ((q & mask) - 2 ** (quantize_bit - 1)).to(x.dtype)
        q = q >> quantize_bit
    s = s.unsqueeze(-2)
    return x * s + o + l @ r


if __name__ == '__main__':
    M, R = 32, 16
    B = 1
    quantize_bit = 4
    element_num = 8 // quantize_bit
    l = torch.randn(M, R, device='cuda', dtype=torch.bfloat16)
    r = torch.randn(R, M, device='cuda', dtype=torch.bfloat16)
    q = torch.randint(0, 255, (B, M, M // element_num), device='cuda', dtype=torch.uint8)
    s = torch.zeros(B, M, device='cuda', dtype=torch.bfloat16) + 2
    o = torch.randn(B, M, M, device='cuda', dtype=torch.bfloat16)
    x = low_rank_addition_fuse_decompression_dequantization(l, r, q, o, s, quantize_bit=quantize_bit)
    q = q.to(torch.int8)
    x_base = torch_low_rank_addition_fuse_decompression_dequantization(l, r, q, o, s, quantize_bit=quantize_bit)
    x_diff = x - x_base
    print(x_diff)
    print(torch.sum(x_diff == 0).item() / x_diff.numel())