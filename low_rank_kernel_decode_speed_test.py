import torch
import triton
import triton.language as tl

from low_rank_extraction_speed import low_rank_addition, torch_low_rank_addition
from low_rank_extraction_fuse_decompression_dequantization import low_rank_addition_fuse_decompression_dequantization, torch_low_rank_addition_fuse_decompression_dequantization

configs = []
for fp8_inputs in [False]:
    configs.append(
        triton.testing.Benchmark(
            x_names=["M"],  # Argument names to use as an x-axis for the plot
            x_vals=[256 * i for i in range(2, 100, 10)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["cuBLAS".lower(), "cublas-w-tail", "triton-w/o-tail", "triton-w-tail"],  # Label name for the lines
            line_names=["cublas", "cublas-w-Tail", "Triton-w/o-Tail", "Triton-w-Tail"],  # Line styles
            styles=[("green", "-"), ("blue", "-"), ("red", "-"), ("black", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        )
    )


@triton.testing.perf_report(configs)
def benchmark(M, provider, fp8_inputs):
    B = 4
    R = 16
    quantize_bit = 2
    element_num = 8 // quantize_bit
    x = torch.randn((B, M, M), device="cuda", dtype=torch.bfloat16)
    l = torch.randn((M, R), device="cuda", dtype=torch.bfloat16)
    r = torch.randn((R, M), device="cuda", dtype=torch.bfloat16)
    s = torch.randn((B, M), device="cuda", dtype=torch.bfloat16) + 2
    o = torch.randn((B, M, M), device=x.device, dtype=torch.bfloat16)
    q = torch.zeros((B, M, M // element_num), device=x.device, dtype=torch.uint8)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "cublas":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_low_rank_addition(l, r, x), quantiles=quantiles)
    if provider == 'cublas-w-tail':
        q = q.to(torch.int8)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_low_rank_addition_fuse_decompression_dequantization(l, r, q, o, s, quantize_bit), quantiles=quantiles)
    if provider == 'triton-w/o-tail':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: low_rank_addition(l, r, x), quantiles=quantiles)
    if provider == 'triton-w-tail':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: low_rank_addition_fuse_decompression_dequantization(l, r, q, o, s, quantize_bit), quantiles=quantiles)
    perf = lambda ms: 2 * B * M * R * M * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=False, print_data=True, save_path="./task3")