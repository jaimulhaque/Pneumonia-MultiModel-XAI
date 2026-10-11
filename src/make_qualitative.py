from PIL import Image
Image.init()
from pathlib import Path
F = Path("results/figures"); OUT = Path("results/paper_figs"); OUT.mkdir(exist_ok=True)
GAP = 20

def crop(im, x0, x1, y0, y1):
    W, H = im.size
    return im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))

def hcat(parts):
    h = max(p.height for p in parts); w = sum(p.width for p in parts) + GAP * (len(parts) - 1)
    c = Image.new("RGB", (w, h), "white"); x = 0
    for p in parts:
        c.paste(p, (x, 0)); x += p.width + GAP
    return c

def vcat(parts):
    w = max(p.width for p in parts); h = sum(p.height for p in parts) + GAP * (len(parts) - 1)
    c = Image.new("RGB", (w, h), "white"); y = 0
    for p in parts:
        c.paste(p, (0, y)); y += p.height + GAP
    return c

# (a) correct cases: original | EffNetV2-S Grad-CAM | +ECA Grad-CAM
im = Image.open(F / "xai_correct_cases.png").convert("RGB")
cols = [(0.000, 0.195), (0.200, 0.396), (0.604, 0.800)]
a = hcat([crop(im, x0, x1, 0, 1) for x0, x1 in cols])
a.save(OUT / "xai_qualitative_correct.png"); a.save(OUT / "xai_qualitative_correct.pdf", resolution=300)

# (b) errors: first row of each figure, original | +ECA Grad-CAM
rows = []
for f in ["xai_errors_bacteria_to_virus", "xai_errors_virus_to_bacteria"]:
    e = Image.open(F / f"{f}.png").convert("RGB")
    rows.append(crop(e, 0.000, 0.660, 0.045, 0.282))
b = vcat(rows)
b.save(OUT / "xai_qualitative_errors.png"); b.save(OUT / "xai_qualitative_errors.pdf", resolution=300)
print("DONE", a.size, b.size)
