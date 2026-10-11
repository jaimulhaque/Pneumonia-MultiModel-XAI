import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path("/mnt/g/Research paper/Research paper/Pneumonia-MultiModel-XAI")
sys.path.insert(0, str(ROOT / "src"))
from pneumo_models import build_model

SPECS = {
    "EffNetV2-S": "baseline:efficientnet_v2_s",
    "EffNetV2-S + ECA": "proposed:efficientnet_v2_s:eca",
    "EffNetV2-S + CBAM": "proposed:efficientnet_v2_s:cbam",
    "EffNet-B2": "baseline:efficientnet_b2",
    "ResNet-50": "baseline:resnet50",
    "ConvNeXt-T": "baseline:convnext_tiny",
    "Swin-T": "baseline:swin_t",
}
SIZE, WARM, ITERS = 384, 30, 100
dev = torch.device("cuda")

def lat(model, x, amp):
    ts = []
    with torch.no_grad(), torch.autocast("cuda", enabled=amp):
        for _ in range(WARM): model(x)
        torch.cuda.synchronize()
        for _ in range(ITERS):
            s = time.perf_counter(); model(x); torch.cuda.synchronize(); ts.append((time.perf_counter() - s) * 1e3)
    return float(np.median(ts)), float(np.std(ts))

rows = []
for name, spec in SPECS.items():
    try:
        m = build_model(spec, pretrained=False).to(dev).eval()
        params = sum(p.numel() for p in m.parameters()) / 1e6
        x1 = torch.randn(1, 3, SIZE, SIZE, device=dev)
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(x1)
        gmacs = fc.get_total_flops() / 2 / 1e9
        l32, s32 = lat(m, x1, False)
        l16, s16 = lat(m, x1, True)
        x16 = torch.randn(16, 3, SIZE, SIZE, device=dev)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad(), torch.autocast("cuda"): m(x16)
        mem = torch.cuda.max_memory_allocated() / 2**20
        rows.append(dict(Model=name, Params_M=round(params, 2), GMACs=round(gmacs, 2),
                         Lat_fp32_ms=round(l32, 2), Lat_fp32_sd=round(s32, 2),
                         Lat_amp_ms=round(l16, 2), Lat_amp_sd=round(s16, 2),
                         Infer_mem_bs16_MB=round(mem)))
        print(rows[-1], flush=True)
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(name, "FAILED:", repr(e)[:200], flush=True)

df = pd.DataFrame(rows)
df.to_csv(ROOT / "results" / "complexity.csv", index=False)
print(df.to_string(index=False))
