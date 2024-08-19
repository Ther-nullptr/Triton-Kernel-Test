import torch
import bitsandbytes as bnb
from low_rank_extraction_fuse_compression_quantization import low_rank_addition_fuse_compression_quantization
from low_rank_extraction_fuse_decompression_dequantization import low_rank_addition_fuse_decompression_dequantization
from torch.profiler import profile, ProfilerActivity
import bitsandbytes.functional as F

def compute_lora(x, w, a, b):
    y_main = w(x)
    y_low_rank = (x @ a) @ b
    y = y_main + y_low_rank
    return y
    
def compute_lora_gradient(dy, x, w, a, b):
    dx = dy @ (w.mT + b.mT @ a.mT)
    da = x.mT @ (dy @ b.mT)
    db = (x @ a).mT @ dy
    return dx, da, db
    
if __name__ == '__main__':
    d1 = 4096
    d2 = 14336
    s = 512
    r = 16
    # initialize the buffered activation
    x = torch.randn((4, s, d1)).cuda().to(torch.bfloat16)
    dy = torch.randn((4, s, d2)).cuda().to(torch.bfloat16)
    scale = torch.randn((4, d1)).cuda().to(torch.bfloat16) + 2
    
    fp16_w = torch.nn.Linear(d1, d2)
    nf4_w = bnb.nn.LinearNF4(d1, d2, compute_dtype=torch.bfloat16)
    nf4_w.load_state_dict(fp16_w.state_dict())
    nf4_w = nf4_w.cuda()

    L = torch.randn((d1, r)).cuda().to(torch.bfloat16)
    R = torch.randn((r, d2)).cuda().to(torch.bfloat16)
    
    L_x = torch.randn((s, r)).cuda().to(torch.bfloat16)
    R_x = torch.randn((r, d1)).cuda().to(torch.bfloat16)
    
    # prehot
    for _ in range(10):
        y = compute_lora(x, nf4_w, L, R)
        
    outlier_val = torch.kthvalue((x[0] - L_x @ R_x).flatten().abs().to(torch.float32), int(x[0].numel() * (1 - 0.01))).values.item()
        
    # profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(100):
            # forward
            y = compute_lora(x, nf4_w, L, R)
            o, q = low_rank_addition_fuse_compression_quantization(L_x, R_x, x, scale, outlier=outlier_val, quantize_bit=2)

    # print(q)
    prof.export_chrome_trace("trace3_A800_2bit.json")
