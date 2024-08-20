import torch
import triton
import triton.language as tl

'''SiLU
    sigmoid = F.sigmoid(x)
    grad_input = sigmoid * (1 + x - x * sigmoid) * grad_output
'''

def torch_silu_backward(x, grad_output):
    sigmoid = torch.sigmoid(x)
    grad_input = sigmoid * (1 + x - x * sigmoid) * grad_output
    return grad_input


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8, }, num_stages=2, 
                      num_warps=2), #! config: 1bit, 2; 2bit, 2; 4bit, 4; 8bit, 4
    ],
    key=['M', 'N'],
)
@triton.jit
def triton_silu_backward_kernel(
    # Pointers to matrices
    x_ptr, grad_output_ptr, grad_input_ptr,
    # Matrix dimensions
    B, M, N,
    # The stride variables
    stride_b, stride_m, stride_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
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
    
    # create pointers
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    x_ptrs = x_ptr + (offs_b * stride_b + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)
    grad_output_ptrs = grad_output_ptr + (offs_b * stride_b + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)
    
    # tiling hadamard
    grad_input = tl.zeros(shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.bfloat16)
    x = tl.load(x_ptrs)
    grad_output = tl.load(grad_output_ptrs)
    sigmoid = tl.sigmoid(x.to(tl.float32)).to(tl.bfloat16)
    grad_input = sigmoid * (1 + x - x * sigmoid) * grad_output
    
    # write back
    grad_input_ptrs = grad_input_ptr + (offs_b * stride_b + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)
    grad_input_mask = (offs_b < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(grad_input_ptrs, grad_input, mask=grad_input_mask)
    
    
def triton_silu_backward(x, grad_output):
    B, M, N = x.shape
    grad_input = torch.empty_like(x)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    triton_silu_backward_kernel[grid](
        x, grad_output, grad_input,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
    )
    return grad_input


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8, }, num_stages=2, 
                      num_warps=2), #! config: 1bit, 2; 2bit, 2; 4bit, 4; 8bit, 4
    ],
    key=['M', 'N'],
)
@triton.jit
def triton_silu_backward_fuse_decompress_kernel(
    # Pointers to matrices
    l_ptr, r_ptr, x_ptr, y_ptr, x_temp_ptr, q_ptr, s_ptr,
    # Matrix dimensions
    B, M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_lm` is how much to increase `l_ptr`
    # by to get the element one row down (A has M rows).
    stride_lm, stride_lk,
    stride_rk, stride_rn,
    stride_xb, stride_xm, stride_xn,
    stride_yb, stride_ym, stride_yn,
    stride_x_tempb, stride_x_tempm, stride_x_tempn,
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

    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    l_ptrs = l_ptr + (offs_m[:, None] * stride_lm + offs_k[None, :] * stride_lk)
    r_ptrs = r_ptr + (offs_k[:, None] * stride_rk + offs_n[None, :] * stride_rn)

    x_ptrs = x_ptr + stride_xm * offs_m[:, None] + stride_xn * offs_n[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :] + offs_b * stride_yb
    y_mask = (offs_b < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # quantization operators
    offs_qm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_qn = pid_n * (BLOCK_SIZE_N // elem_per_position) + tl.arange(0, BLOCK_SIZE_N // elem_per_position)
    q_ptrs = q_ptr + stride_qm * offs_qm[:, None] + stride_qn * offs_qn[None, :] + offs_b * stride_qb
    q_mask = (offs_b < B) & (offs_qm[:, None] < M) & (offs_qn[None, :] < N // elem_per_position)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    offs_x_temp_m = offs_m
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
    s_ptrs = s_ptr + stride_sn * offs_sn[None, :] 
    s_mask = (offs_sn[None, :] < N)
    s = tl.load(s_ptrs, mask=s_mask, other=1.0)
    
    offs_x_temp_m = offs_m
    offs_x_temp_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_temp_ptrs = x_temp_ptr + stride_xm * offs_x_temp_m[:, None] + stride_xn * offs_x_temp_n[None, :] + offs_b * stride_xb
    
    x = tl.load(x_temp_ptrs, mask=x_mask, other=0.0)
    x = x - 2 ** (quantize_bit - 1)
    x = x.to(tl.bfloat16)
    x = x * s

    accumulator = x
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
        
    # load y
    y = tl.load(y_ptrs, mask=y_mask, other=0.0)
    y = accumulator.to(tl.bfloat16) * y
    
    #  tiling hadamard
    sigmoid = tl.sigmoid(accumulator.to(tl.float32)).to(tl.bfloat16)
    y = sigmoid * (1 + accumulator - accumulator * sigmoid) * y

    tl.store(x_ptrs, y, mask=x_mask)


def triton_silu_backward_fuse_extraction(l, r, q, s, b, quantize_bit=2):
    M, K = l.shape
    K, N = r.shape
    B, _, _ = q.shape
    
    elem_per_position = 8 // quantize_bit
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    c = torch.empty((B, M, N), device=l.device, dtype=torch.bfloat16)
    a_temp = torch.empty((B, M, N), device=l.device, dtype=torch.uint8)
    triton_silu_backward_fuse_decompress_kernel[grid](
        l, r, c, b, a_temp, q, s,
        B, M, N, K,
        l.stride(0), l.stride(1),
        r.stride(0), r.stride(1),
        c.stride(0), c.stride(1), c.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        a_temp.stride(0), a_temp.stride(1), a_temp.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(0), s.stride(1),
        quantize_bit, elem_per_position,
        BLOCK_SIZE_K=K
    )
    return c


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8, }, num_stages=2, 
                      num_warps=2), #! config: 1bit, 2; 2bit, 2; 4bit, 4; 8bit, 4
    ],
    key=['M', 'N'],
)
@triton.jit
def low_rank_addition_fuse_decompression_dequantization_kernel(
    # Pointers to matrices
    l_ptr, r_ptr, x_ptr, x_temp_ptr, q_ptr, s_ptr,
    # Matrix dimensions
    B, M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_lm` is how much to increase `l_ptr`
    # by to get the element one row down (A has M rows).
    stride_lm, stride_lk,
    stride_rk, stride_rn,
    stride_xb, stride_xm, stride_xn,
    stride_x_tempb, stride_x_tempm, stride_x_tempn,
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
    s_ptrs = s_ptr + stride_sn * offs_sn[None, :] 
    s_mask = (offs_sn[None, :] < N)
    s = tl.load(s_ptrs, mask=s_mask, other=1.0)
    
    offs_x_temp_m = offs_xm
    offs_x_temp_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_temp_ptrs = x_temp_ptr + stride_xm * offs_x_temp_m[:, None] + stride_xn * offs_x_temp_n[None, :] + offs_b * stride_xb
    
    x = tl.load(x_temp_ptrs, mask=x_mask, other=0.0)
    x = x - 2 ** (quantize_bit - 1)
    x = x.to(tl.bfloat16)
    x = x * s
    
    accumulator = tl.zeros(shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
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


def triton_silu_backward_add_extraction(l, r, q, s, b, quantize_bit=2):
    M, K = l.shape
    K, N = r.shape
    B, _, _ = q.shape
    elem_per_position = 8 // quantize_bit
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    a = torch.empty((B, M, N), device=l.device, dtype=torch.bfloat16)
    a_temp = torch.empty((B, M, N), device=l.device, dtype=torch.uint8)
    c = torch.empty((B, M, N), device=l.device, dtype=torch.bfloat16)
    low_rank_addition_fuse_decompression_dequantization_kernel[grid](
        l, r, a, a_temp, q, s,
        B, M, N, K,
        l.stride(0), l.stride(1),
        r.stride(0), r.stride(1),
        a.stride(0), a.stride(1), a.stride(2),
        a_temp.stride(0), a_temp.stride(1), a_temp.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        s.stride(0), s.stride(1),
        quantize_bit, elem_per_position,
        BLOCK_SIZE_K=K
    )
    triton_silu_backward_kernel[grid](
        a, b, c,
        B, M, N,
        a.stride(0), a.stride(1), a.stride(2),
    )
    

# test accuracy
a = torch.randn((4, 1024, 1024), device="cuda", dtype=torch.bfloat16)
b = torch.randn((4, 1024, 1024), device="cuda", dtype=torch.bfloat16)
c = torch_silu_backward(a, b)
triton_c = triton_silu_backward(a, b)
if torch.allclose(c, triton_c):
    print("✅ Triton and Torch match")
    print(f"c: {c}")
    print(f"triton_c: {triton_c}")
    print(f"Results are different!, error: {torch.norm(c - triton_c).item()}")
else:
    print(f"c: {c}")
    print(f"triton_c: {triton_c}")
    print(f"Results are different!, error: {torch.norm(c - triton_c).item()}")
    
    
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N'],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cublas', 'triton', 'triton-fuse', 'triton-no-fuse'],
        # Label name for the lines
        line_names=["cuBLAS", "Triton", "triton-fuse", "triton-no-fuse"],
        # Line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('purple', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(M, N, provider):
    B = 4
    R = 16
    quantize_bit = 2
    element_num = 8 // quantize_bit
    
    a = torch.randn((B, M, N), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((B, M, N), device="cuda", dtype=torch.bfloat16)
    l = torch.randn((M, R), device="cuda", dtype=torch.bfloat16)
    r = torch.randn((R, M), device="cuda", dtype=torch.bfloat16)
    s = torch.randn((B, M), device="cuda", dtype=torch.bfloat16) + 2
    q = torch.randint(0, 255, (B, M, N // element_num), device="cuda", dtype=torch.uint8) 
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_silu_backward(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_silu_backward(a, b), quantiles=quantiles)
    if provider == 'triton-fuse':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_silu_backward_fuse_extraction(l, r, q, s, b, quantize_bit=quantize_bit), quantiles=quantiles)
    if provider == 'triton-no-fuse':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_silu_backward_add_extraction(l, r, q, s, b, quantize_bit=quantize_bit), quantiles=quantiles)
    perf = lambda ms: 2 * B * M * N * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(print_data=True, show_plots=False, save_path="./silu_backward")