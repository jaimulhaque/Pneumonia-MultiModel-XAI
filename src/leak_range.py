import numpy as np
from pathlib import Path
R = Path("results"); tag = "baseline_efficientnet_v2_s"; d = []
for k in range(5):
    acc = {}
    for name, base in [("img", R / "leakage_img"), ("pat", R)]:
        z = np.load(base / f"fold{k}_seed42" / tag / "test_probs.npz")
        y, p = z["y"], z["p"]
        acc[name] = 100 * float((p.argmax(1) == y).mean())
    d.append(acc["img"] - acc["pat"])
    print(f"fold {k}: image {acc['img']:.2f}  patient {acc['pat']:.2f}  inflation {d[-1]:+.2f}")
print(f"min {min(d):+.2f}  max {max(d):+.2f}  mean {np.mean(d):+.2f}")
