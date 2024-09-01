import torch
import triton
import triton.language as tl


def torch_softmax_backward(y, grad_y):
    return (grad_y - (grad_y * y).sum(dim=-1, keepdims=True)) * y


@triton.jit
def triton_softmax_backward_kernel(
    y_ptr, g_ptr, o_ptr,
    b_stride, h_stride, row_stride, col_stride,
    b, h, n_rows, n_cols, 
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs_b = tl.program_id(axis=1)
    offs_h = tl.program_id(axis=2)
    num_pid_m = tl.cdiv(n_rows, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(n_cols, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    y_row_ptr = y_ptr + row_stride * offs_m[:, None] + col_stride * offs_n[None, :] + b_stride * offs_b + h_stride * offs_h
    g_row_ptr = g_ptr + row_stride * offs_m[:, None] + col_stride * offs_n[None, :] + b_stride * offs_b + h_stride * offs_h
    mask = (offs_b < b) & (offs_h < h) & (offs_m[:, None] < n_rows) & (offs_n[None, :] < n_cols) 
    
    y = tl.load(y_row_ptr, mask=mask)
    g = tl.load(g_row_ptr, mask=mask)
    y_g = y * g
    y_g_sum = tl.sum(y_g)
    o = (g - y_g_sum) * y
    
    o_row_ptr = o_ptr + row_stride * offs_m[:, None] + col_stride * offs_n[None, :] + b_stride * offs_b + h_stride * offs_h
    tl.store(o_row_ptr, o, mask=mask)

        
def triton_softmax_backward(y, grad_y):    
    b, h, n, _ = y.shape
    o = torch.empty_like(y)
    
    assert grad_y.shape == y.shape
    grid = lambda META: (
        triton.cdiv(n, META['BLOCK_SIZE_M']) * triton.cdiv(n, META['BLOCK_SIZE_N']), b, h
    )
    # Create a number of persistent programs.
    triton_softmax_backward_kernel[grid](
        y, grad_y, o, 
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        b, h, n, n, 
        BLOCK_SIZE_M=1, BLOCK_SIZE_N=n, GROUP_SIZE_M=1
    )
    return o

x = torch.randn(2, 2, 512, 512).to(torch.bfloat16).cuda()
y = torch.softmax(x, dim=-1).to(torch.bfloat16)
g = torch.randn_like(y).to(torch.bfloat16).cuda()
o_torch = torch_softmax_backward(y, g)
o_triton = triton_softmax_backward(y, g)

if torch.allclose(o_torch, o_triton, rtol=1e-2, atol=1e-2):
    print('Accuracy test passed!')
    print(o_torch - o_triton)
else:
    print('Accuracy test failed!')
    print(o_torch - o_triton)
    

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
    x = torch.randn(B, M, N, device='cuda', dtype=torch.bfloat16)
    y = torch.softmax(x, dim=-1)
    g = torch.randn_like(y)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_softmax_backward(y, g), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_softmax_backward(y, g), quantiles=quantiles)
        
    perf = lambda ms: 2 * B * M * N * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(print_data=True, show_plots=False, save_path='./softmax_backward')