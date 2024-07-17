import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8, }, num_stages=1, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def low_rank_addition_kernel(
    # Pointers to matrices
    l_ptr, r_ptr, x_ptr, o_ptr,
    # Matrix dimensions
    B, M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_lm` is how much to increase `l_ptr`
    # by to get the element one row down (A has M rows).
    stride_lm, stride_lk,
    stride_rk, stride_rn,
    stride_xb, stride_xm, stride_xn,
    stride_ob, stride_om, stride_on,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, 
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
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
    o_mask = (offs_b < B) & (offs_om[:, None] < M) & (offs_on[None, :] < N)
    
    accumulator = tl.load(x_ptrs, mask=x_mask, other=0.0)
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

    # divide the simple values and outlier values
    x = accumulator.to(tl.bfloat16)
    tl.store(o_ptrs, x, mask=o_mask)


def low_rank_addition(l, r, x):
    # Check constraints.
    assert l.shape[1] == r.shape[0], "Incompatible dimensions"
    assert l.is_contiguous(), "Matrix A must be contiguous"
    assert r.is_contiguous(), "Matrix B must be contiguous"
    M, K = l.shape
    K, N = r.shape
    B, _, _ = x.shape
    if K < 16:
        l = torch.cat([l, torch.zeros((M, 16 - K), device=l.device, dtype=l.dtype)], dim=1).contiguous()
        r = torch.cat([r, torch.zeros((16 - K, N), device=r.device, dtype=r.dtype)], dim=0).contiguous()
        K = 16
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    o = torch.empty((B, M, N), device=x.device, dtype=torch.bfloat16)
    low_rank_addition_kernel[grid](
        l, r, x, o,
        B, M, N, K,
        l.stride(0), l.stride(1),
        r.stride(0), r.stride(1),
        x.stride(0), x.stride(1), x.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        BLOCK_SIZE_K=K
    )
    return o


def torch_low_rank_addition(l, r, x):
    return x + l @ r


def torch_low_rank_addition_fuse_compression_quantization(l, r, x, s, quantize_bit=8, outlier=5.):
    x = x - l @ r
    # step 1: extract the outlier
    outlier_mask = torch.abs(x) > outlier
    o = x * outlier_mask
    x = x - o
    # step 2: compress the rest part
    q = torch.clamp(torch.round(x / s), min = -2 ** (quantize_bit-1), max = 2 ** (quantize_bit-1) - 1).to(torch.int32)
    
    x = q.to(torch.bfloat16) * s + o + l @ r
    return o, q

# configs = []
# for fp8_inputs in [False]:
#     configs.append(
#         triton.testing.Benchmark(
#             x_names=["M"],  # Argument names to use as an x-axis for the plot
#             x_vals=[256 * i for i in range(2, 100, 2)],  # Different possible values for `x_name`
#             line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
#             # Possible values for `line_arg`
#             # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
#             line_vals=["triton"] if fp8_inputs else ["cuBLAS".lower(), "triton"],  # Label name for the lines
#             line_names=["Triton"] if fp8_inputs else ["cublas", "Triton"],  # Line styles
#             styles=[("green", "-"), ("blue", "-")],
#             ylabel="TFLOPS",  # Label name for the y-axis
#             plot_name="matmul-performance-" +
#             ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
#             args={"fp8_inputs": fp8_inputs},
#         ))


# @triton.testing.perf_report(configs)
# def benchmark(M, provider, fp8_inputs):
#     B = 4
#     R = 16
#     x = torch.randn((B, M, M), device="cuda", dtype=torch.bfloat16)
#     l = torch.randn((M, R), device="cuda", dtype=torch.bfloat16)
#     r = torch.randn((R, M), device="cuda", dtype=torch.bfloat16)
#     quantiles = [0.5, 0.2, 0.8]
#     if provider == "cublas":
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_low_rank_addition(l, r, x), quantiles=quantiles)
#     if provider == 'triton':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: low_rank_addition(l, r, x), quantiles=quantiles)
#     perf = lambda ms: 2 * B * M * R * M * 1e-12 / (ms * 1e-3)
#     return perf(ms), perf(max_ms), perf(min_ms)


# benchmark.run(show_plots=False, print_data=True, save_path="./task1")