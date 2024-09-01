import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from packaging.version import Version

triton_version = triton.__version__
if Version(triton_version) < Version("3.0.0"):
    import triton.language.math as tl_math
else:
    import triton.language.extra.cuda.libdevice as tl_math


def torch_gelu_backward(x, grad_y):
    gamma = 0.7978845608028654 # math.sqrt(2 / math.pi)
    kappa = 0.044715
    y = gamma * (x + kappa * x**3)
    tanh_y = F.tanh(y)
    grad_input = (0.5 * ((1 + tanh_y) + x * ((1 - tanh_y**2) * gamma * (1 + 3 * kappa * x**2))) * grad_y)
    
    return grad_input


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE_M': 8, }, num_stages=2, num_warps=2), #! config: 1bit, 2; 2bit, 2; 4bit, 4; 8bit, 4
    ],
    key=['M', 'N'],
)
@triton.jit
def triton_gelu_backward_kernel(
    x_ptr, g_ptr, o_ptr,
    B, M, N,
    stride_xb, stride_xm, stride_xn,
    stride_gb, stride_gm, stride_gn,
    stride_ob, stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_ptrs = x_ptr + stride_xm * offs_m[:, None] + stride_xn * offs_n[None, :] + offs_b * stride_xb
    x_mask = (offs_b < B) & (offs_m[:, None] < M) & (offs_n[None, :] < N)
    g_ptrs = g_ptr + stride_gm * offs_m[:, None] + stride_gn * offs_n[None, :] + offs_b * stride_gb
    g_mask = x_mask
    o_ptrs = o_ptr + stride_om * offs_m[:, None] + stride_on * offs_n[None, :] + offs_b * stride_ob
    o_mask = x_mask
    
    x = tl.load(x_ptrs, mask=x_mask)
    g = tl.load(g_ptrs, mask=g_mask)
    gamma = 0.7978845608028654 # math.sqrt(2 / math.pi)
    kappa = 0.044715
    x_pow_2 = x * x
    x_pow_3 = x_pow_2 * x
    y = gamma * (x + kappa * x_pow_3)
    tanh_y = tl_math.tanh(y)
    tanh_y_pow_2 = tanh_y * tanh_y
    o = (0.5 * ((1 + tanh_y) + x * ((1 - tanh_y_pow_2) * gamma * (1 + 3 * kappa * x_pow_2))) * g)
    
    tl.store(o_ptrs, o, mask=o_mask)
    

def triton_gelu_backward(x, g):
    B, M, N = x.shape
    assert g.shape == x.shape
    
    o = torch.empty_like(x)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), B
    )
    triton_gelu_backward_kernel[grid](
        x, g, o,
        B, M, N,
        x.stride(0), x.stride(1), x.stride(2),
        g.stride(0), g.stride(1), g.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
    )
    return o


# test accuracy
x = torch.randn(4, 1024, 1024, device='cuda', dtype=torch.bfloat16)
g = torch.randn_like(x)
o_torch = torch_gelu_backward(x, g)
o_triton = triton_gelu_backward(x, g)

if torch.allclose(o_torch, o_triton, rtol=1e-2, atol=1e-2):
    print('Accuracy test passed!')
    print(o_torch)
    print(o_triton)
else:
    print('Accuracy test failed!')
    print(o_torch)
    print(o_triton)
    
    
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N'],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cublas', 'triton'],
        # Label name for the lines
        line_names=["cuBLAS", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    )
)
def benchmark(M, N, provider):
    B = 4
    x = torch.randn((B, M, N), device='cuda', dtype=torch.bfloat16)
    g = torch.randn_like(x)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_gelu_backward(x, g), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_gelu_backward(x, g), quantiles=quantiles)
        
    perf = lambda ms: 2 * B * M * N * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)
        

benchmark.run(print_data=True, show_plots=False, save_path='./gelu_backward')