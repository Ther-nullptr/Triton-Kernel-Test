import torch
import triton
import triton.language as tl

def torch_extract_unstructured(x, threshold):
    x_outlier = x * (torch.abs(x) > threshold)
    x_no_outlier = x - x_outlier
    return x_outlier.to_sparse(), x_no_outlier


def torch_extract_channel(x, channel_index):
    x_outlier_channel = x[:, channel_index, :]
    x[:, channel_index, :] = 0
    return x_outlier_channel, x


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N'],
        x_vals=[
            256 * i for i in range(2, 33)
        ],
        line_arg='provider',
        line_vals=['unstructrued', 'channel'],
        line_names=['unstructrued', 'channel'],
        styles=[('red', '-'), ('blue', '-')],
        ylabel='TFLOPS',
        plot_name='extraction',
        args={}
    )
)
def benchmark(M, N, provider):
    B = 4
    a = torch.randn((B, M, N), device='cuda', dtype=torch.bfloat16)
    # select 5% threshold
    threshold_ratio = 0.05
    threshold = torch.kthvalue(a.flatten().to(torch.float32).abs(), int(a.numel() * (1 - threshold_ratio))).values
    # select 5% channel(randomly)
    channel_ratio = 0.05
    channel_num = int(M * channel_ratio)
    channel_index = torch.randperm(M)[:channel_num]
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'unstructrued':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_extract_unstructured(a, threshold), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_extract_channel(a, channel_index), quantiles=quantiles)
    perf = lambda ms: 2 * B * M * N * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(print_data=True, show_plots=False, save_path='./extraction_outlier')
