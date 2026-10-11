import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from pathlib import Path
D = Path("/mnt/g/Research paper/Research paper/dataset of pneomunia/chest_xray")
sp = pd.read_csv("splits/fold0_test.csv")
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))
for a, c in zip(axes, ["BACTERIA", "NORMAL", "VIRUS"]):
    p = sp[sp["label"].astype(str).str.upper() == c].iloc[3]["image_path"]
    a.imshow(mpimg.imread(str(D / p)), cmap="gray"); a.set_title(c, fontsize=9); a.axis("off")
plt.tight_layout()
Path("overleaf_upload/figs").mkdir(parents=True, exist_ok=True)
plt.savefig("overleaf_upload/figs/fig1_sample_classes.jpg", dpi=300)
print("DONE")
