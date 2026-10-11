"""Generate paper figures/tables from existing results. Run from PROJECT_ROOT:
   python -u src/make_paper_figs.py
Outputs: results/paper_figs/*.png and results/paper_figs/table_perclass_body.tex"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score, roc_auc_score

ROOT = Path("/mnt/g/Research paper/Research paper/Pneumonia-MultiModel-XAI")
RES = ROOT / "results"; OUT = RES / "paper_figs"; OUT.mkdir(exist_ok=True)
CLASSES = ["NORMAL", "BACTERIA", "VIRUS"]
TAGS = {
    "EffNetV2-S": "baseline_efficientnet_v2_s",
    "+ ECA": "proposed_efficientnet_v2_s_eca",
    "+ CBAM": "proposed_efficientnet_v2_s_cbam",
    "EffNet-B2": "baseline_efficientnet_b2",
    "ResNet-50": "baseline_resnet50",
    "ConvNeXt-T": "baseline_convnext_tiny",
    "Swin-T": "baseline_swin_t",
}

def load(tag):
    ys, ps, fs = [], [], []
    for k in range(5):
        d = np.load(RES / f"fold{k}_seed42" / tag / "test_probs.npz")
        ys.append(d["y"]); ps.append(d["p"]); fs.append(np.full(len(d["y"]), k))
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(fs)

DATA = {n: load(t) for n, t in TAGS.items()}

# ---- Table: pooled per-class F1, accuracy, macro-F1, macro-AUC (LaTeX body) ----
rows = []
for n, (y, p, f) in DATA.items():
    pred = p.argmax(1)
    f1c = f1_score(y, pred, average=None, labels=[0, 1, 2])
    rows.append((n, (pred == y).mean() * 100, f1_score(y, pred, average="macro"),
                 *f1c, roc_auc_score(y, p, multi_class="ovr", average="macro")))
df = pd.DataFrame(rows, columns=["Model", "Acc", "MacroF1", "F1_NORMAL", "F1_BACTERIA", "F1_VIRUS", "MacroAUC"])
df.to_csv(OUT / "table_perclass.csv", index=False)
with open(OUT / "table_perclass_body.tex", "w") as fh:
    fh.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
    fh.write("Model & Acc (\\%) & Macro-F1 & F1 NORMAL & F1 BACT. & F1 VIRUS & Macro-AUC\\\\\n\\midrule\n")
    for r in df.itertuples(index=False):
        fh.write(f"{r.Model} & {r.Acc:.2f} & {r.MacroF1:.3f} & {r.F1_NORMAL:.3f} & {r.F1_BACTERIA:.3f} & {r.F1_VIRUS:.3f} & {r.MacroAUC:.3f}\\\\\n")
    fh.write("\\bottomrule\n\\end{tabular}\n")
print(df.round(4).to_string(index=False))

# ---- Fig: per-fold accuracy ----
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for i, (n, (y, p, f)) in enumerate(DATA.items()):
    accs = [(p[f == k].argmax(1) == y[f == k]).mean() * 100 for k in range(5)]
    ax.bar(i, np.mean(accs), color="#cfd8e3", edgecolor="k", lw=0.5)
    ax.scatter([i] * 5, accs, color="#1f4e79", s=14, zorder=3)
ax.set_xticks(range(len(DATA))); ax.set_xticklabels(DATA.keys(), rotation=25, ha="right")
ax.set_ylabel("Test accuracy (%)"); ax.set_ylim(78, 88)
plt.tight_layout(); plt.savefig(OUT / "fig_perfold_acc.png", dpi=300); plt.close()

# ---- Fig: pooled confusion matrices ----
sel = ["EffNetV2-S", "+ ECA", "Swin-T"]
fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
for ax, n in zip(axes, sel):
    y, p, _ = DATA[n]
    cm = confusion_matrix(y, p.argmax(1), labels=[0, 1, 2])
    cmn = cm / cm.sum(1, keepdims=True) * 100
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.1f}%", ha="center", va="center",
                    color="white" if cmn[i, j] > 55 else "black", fontsize=8)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES, fontsize=8); ax.set_yticklabels(CLASSES, fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(n)
plt.tight_layout(); plt.savefig(OUT / "fig_confusion_pooled.png", dpi=300); plt.close()

# ---- Fig: pooled one-vs-rest ROC (EffNetV2-S, +ECA) ----
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)
for ax, n in zip(axes, ["EffNetV2-S", "+ ECA"]):
    y, p, _ = DATA[n]
    for c, cname in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y == c, p[:, c])
        ax.plot(fpr, tpr, label=f"{cname} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=0.7)
    ax.set_title(n); ax.set_xlabel("False positive rate"); ax.legend(fontsize=7, loc="lower right")
axes[0].set_ylabel("True positive rate")
plt.tight_layout(); plt.savefig(OUT / "fig_roc_pooled.png", dpi=300); plt.close()

# ---- Fig: training curves, fold 0 ----
fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))
for n in ["EffNetV2-S", "+ ECA"]:
    h = pd.read_csv(RES / "fold0_seed42" / TAGS[n] / "history.csv")
    lc = [c for c in h.columns if "loss" in c.lower()]
    ac = [c for c in h.columns if "acc" in c.lower()]
    for c in lc: axes[0].plot(h[c].values, label=f"{n}: {c}")
    for c in ac: axes[1].plot(h[c].values, label=f"{n}: {c}")
axes[0].set_title("Loss"); axes[1].set_title("Accuracy")
for ax in axes: ax.set_xlabel("Epoch"); ax.legend(fontsize=6)
plt.tight_layout(); plt.savefig(OUT / "fig_training_curves.png", dpi=300); plt.close()

# ---- Fig: leakage inflation ----
lk = pd.read_csv(RES / "leakage_summary.csv")
mets = ["test_acc", "test_macro_f1", "test_mcc", "VIRUS_F1"]
labs = ["Accuracy", "Macro-F1", "MCC", "VIRUS F1"]
fig, ax = plt.subplots(figsize=(6, 3.2)); w = 0.38
for i, (m, nm) in enumerate([("baseline:efficientnet_v2_s", "EffNetV2-S"), ("baseline:resnet50", "ResNet-50")]):
    s = lk[lk.model == m].set_index("metric").loc[mets]
    ax.bar(np.arange(4) + (i - 0.5) * w, s.inflation_pp, w, yerr=s.inflation_sd, capsize=3, label=nm)
ax.set_xticks(range(4)); ax.set_xticklabels(labs); ax.axhline(0, color="k", lw=0.6)
ax.set_ylabel("Inflation (pp)"); ax.legend()
plt.tight_layout(); plt.savefig(OUT / "fig_leakage_inflation.png", dpi=300); plt.close()

# ---- Fig: preprocessing (raw vs CLAHE + pad 384), one image per class from fold 0 test ----
try:
    import cv2
    sp = pd.read_csv(ROOT / "splits" / "fold0_test.csv")
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.4))
    for c, cname in enumerate(CLASSES):
        lab = sp["label"]
        sub = sp[(lab == c) | (lab.astype(str).str.upper() == cname)]
        path = sub.iloc[0]["image_path"]
        im = cv2.imread(str(Path("/mnt/g/Research paper/Research paper/dataset of pneomunia/chest_xray") / str(path)), cv2.IMREAD_GRAYSCALE)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(im)
        s = 384 / max(im.shape); cl = cv2.resize(cl, (int(im.shape[1] * s), int(im.shape[0] * s)))
        pad = np.zeros((384, 384), np.uint8)
        y0, x0 = (384 - cl.shape[0]) // 2, (384 - cl.shape[1]) // 2
        pad[y0:y0 + cl.shape[0], x0:x0 + cl.shape[1]] = cl
        axes[0, c].imshow(im, cmap="gray"); axes[0, c].set_title(cname, fontsize=9)
        axes[1, c].imshow(pad, cmap="gray")
    for a in axes.ravel(): a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "fig_preprocessing.png", dpi=300); plt.close()
except Exception as e:
    print("preprocessing figure skipped:", repr(e)[:200])
print("DONE ->", OUT)
