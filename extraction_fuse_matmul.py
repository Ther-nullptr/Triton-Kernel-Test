import torch
import triton
import triton.language as tl

def torch_matmul(x, a):
    return torch.matmul(x, a)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8, }, num_stages=4, 
                      num_warps=2), #! config: 1bit, 2; 2bit, 2; 4bit, 4; 8bit, 4
    ],
    key=['M', 'N'],
)
@triton.jit
def batched_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1) # [new] dimension for parallel
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_b * stride_ab + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_b * stride_bb + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.bfloat16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :] + offs_b * stride_cb
    c_mask = (offs_b < B) & (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
    
    
def triton_batched_matmul(x, a):
    B, M, K = x.shape
    _, _, N = a.shape
    o = torch.empty((B, M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B 
    )
    batched_matmul_kernel[grid](
        x, a, o,
        B, M, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        a.stride(0), a.stride(1), a.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_SIZE_K=N
    )
    return o


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8, }, num_stages=4, num_warps=2), 
    ],
    key=['M', 'N'],
)
@triton.jit
def triton_split_k_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    B, M, N, R,
    stride_ab, stride_am, stride_an,
    stride_bb, stride_bn, stride_bk,
    stride_cb, stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1) # [new] dimension for parallel
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_an = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = a_ptr + (offs_b * stride_ab + offs_am[:, None] * stride_am + offs_an[None, :] * stride_an)
    a_mask = (offs_am[:, None] < M) & (offs_an[None, :] < N) & (offs_b < B)
    
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_bk = tl.arange(0, BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_b * stride_bb + offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk)
    b_mask = (offs_bn[None, :] < N) & (offs_bk[:, None] < R)
    
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
    accumulator = tl.dot(a, b, out_dtype=tl.float32).to(tl.bfloat16)
    
    offs_cm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_ck = tl.arange(0, BLOCK_SIZE_K)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_ck[None, :] * stride_ck) + offs_b * stride_cb
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < R) & (offs_b < B)
    
    tl.store(c_ptrs, accumulator, mask=c_mask)
    
    
def triton_split_k_matmul(x, a):
    B, M, N = x.shape
    _, _, K = a.shape
    o = torch.zeros((B, M, K), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B 
    )
    triton_split_k_matmul_kernel[grid](
        x, a, o,
        B, M, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        a.stride(0), a.stride(1), a.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_SIZE_K=K
    )
    return o


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, }, num_stages=2, num_warps=2), 
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def triton_matmul_decompress_kernel(
    q_ptr, l_ptr, r_ptr, s_ptr, x_ptr, y_ptr, o_ptr,
    B, M, N, K,
    stride_qb, stride_qm, stride_qk,
    stride_lm, stride_lr,
    stride_rr, stride_rk,
    stride_sb, stride_sk,
    stride_xb, stride_xm, stride_xk,
    stride_yb, stride_yk, stride_yn,
    stride_ob, stride_om, stride_on,
    quantize_bit: tl.constexpr, elem_per_position: tl.constexpr, 
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
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
    
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_qk = tl.arange(0, BLOCK_SIZE_K // elem_per_position)
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    q_ptrs = x_ptr + (offs_m[:, None] * stride_qm + offs_qk[None, :] * stride_qk) + offs_b * stride_xb
    y_ptrs = y_ptr + (offs_k[:, None] * stride_yk + offs_n[None, :] * stride_yn) + offs_b * stride_yb
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_qk[None, :] * stride_xk) + offs_b * stride_xb
    x_decompress_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_yk) + offs_b * stride_xb
    s_ptrs = s_ptr + (offs_k[:, None] * stride_sk) + offs_b * stride_sb
    o_ptrs = o_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on) + offs_b * stride_ob
    l_ptrs = l_ptr + (offs_m[:, None] * stride_lm + offs_k[None, :] * stride_lr)
    r_ptrs = r_ptr + (offs_k[:, None] * stride_rr + offs_k[None, :] * stride_rk)
    
    mask = (1 << quantize_bit) - 1
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        q = tl.load(q_ptrs, mask=offs_qk[None, :] < K - k * (BLOCK_SIZE_K // elem_per_position), other=0.0)
        y = tl.load(y_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        
        # dequantize
        for i in range(elem_per_position):
            x_ptrs_i = x_ptrs + i * (BLOCK_SIZE_K // elem_per_position)
            fake_int = tl.extra.cuda.libdevice.uint2float_rn((q & mask).to(tl.uint32))
            tl.store(x_ptrs_i, fake_int)
            q = (q >> quantize_bit).to(tl.uint8)
            
        s = tl.load(s_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        x = tl.load(x_decompress_ptrs)
        x = x - 2 ** (quantize_bit - 1)
        x = x.to(tl.bfloat16)
        x = x * s
        
        # low-rank addition
        l = tl.load(l_ptrs)
        r = tl.load(r_ptrs)
        x = (x + tl.dot(l, r)).to(tl.bfloat16)
        
        accumulator += (tl.dot(x, y))
        x_ptrs += BLOCK_SIZE_K * stride_xk
        y_ptrs += BLOCK_SIZE_K * stride_yk
        r_ptrs += BLOCK_SIZE_K * stride_rk
        
    o = accumulator.to(tl.bfloat16)
    tl.store(o_ptrs, o)


def triton_matmul_decompress(q, l, r, s, y, quantize_bit=2):
    B, M, K = q.shape
    _, _, N = y.shape
    _, R = r.shape
    
    elem_per_position = 8 // quantize_bit
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    x = torch.zeros((B, M, K), device=q.device, dtype=torch.uint8)
    o = torch.zeros((B, M, N), device=q.device, dtype=torch.bfloat16)
    
    triton_matmul_decompress_kernel[grid](
        q, l, r, s, x, y, o,
        B, M, N, K,
        q.stride(0), q.stride(1), q.stride(2),
        l.stride(0), l.stride(1),
        r.stride(0), r.stride(1),
        s.stride(0), s.stride(1),
        x.stride(0), x.stride(1), x.stride(2),
        y.stride(0), y.stride(1), y.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        quantize_bit, elem_per_position,
    )
    
    return o

# test accuracy
x = torch.randn((32, 1024, 1024), device='cuda', dtype=torch.bfloat16)
a = torch.randn((32, 1024, 32), device='cuda', dtype=torch.bfloat16)

o_torch = torch_matmul(x, a)
o_triton_1 = triton_batched_matmul(x, a)
o_triton_2 = triton_split_k_matmul(x, a)

if torch.allclose(o_torch, o_triton_1):
    print("✅ Triton 1 and Torch match")
else:
    print("❌ Triton 1 and Torch mismatch")
if torch.allclose(o_torch, o_triton_2):
    print("✅ Triton 2 and Torch match")
else:
    print("❌ Triton 2 and Torch mismatch")
print(f"o_torch: {o_torch}")
print(f"o_triton_1: {o_triton_1}")
print(f"o_triton_2: {o_triton_2}")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N'],
        x_vals=[256 * i for i in range(4, 32, 4)],
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['torch', 'triton', 'triton-split', 'triton-decompress'],
        # Label name for the lines
        line_names=['torch', 'triton', 'triton-split', 'triton-decompress'],
        # Line styles
        styles=[('green', '-'), ('blue', '-'), ('red', '-'), ('black', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(M, N, provider):
    B = 4
    R = 32
    quantize_bit = 2
    element_num = 8 // quantize_bit
    
    x = torch.randn((B, M, N), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((B, N, R), device="cuda", dtype=torch.bfloat16) 
    q = torch.randint(0, 255, (B, M, N // element_num), device="cuda", dtype=torch.uint8)
    s = torch.randn((B, M), device="cuda", dtype=torch.bfloat16) + 2
    l = torch.randn((M, R), device="cuda", dtype=torch.bfloat16)
    r = torch.randn((R, M), device="cuda", dtype=torch.bfloat16)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_matmul(x, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_batched_matmul(x, b), quantiles=quantiles)
    if provider == 'triton-split':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_split_k_matmul(x, b), quantiles=quantiles)
    if provider == 'triton-decompress':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul_decompress(q, l, r, s, b, quantize_bit), quantiles=quantiles)
    perf = lambda ms: 2 * B * M * N * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(print_data=True, show_plots=False, save_path="./matmul")