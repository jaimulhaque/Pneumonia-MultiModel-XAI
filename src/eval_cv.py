import json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import binomtest, ttest_rel
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES, SPL = ROOT / "results", ROOT / "splits"
CLASSES = ["NORMAL", "BACTERIA", "VIRUS"]
MODELS = {
    "EffNetV2-S": "baseline_efficientnet_v2_s",
    "+ECA": "proposed_efficientnet_v2_s_eca",
    "+CBAM": "proposed_efficientnet_v2_s_cbam",
    "EffNet-B2": "baseline_efficientnet_b2",
    "ResNet-50": "baseline_resnet50",
    "ConvNeXt-T": "baseline_convnext_tiny",
    "Swin-T": "baseline_swin_t",
}
FOLDS, SEED, B = range(5), 42, 2000
rng = np.random.default_rng(0)

def macro_f1(y, p):
    cm = np.bincount(y * 3 + p, minlength=9).reshape(3, 3)
    tp = np.diag(cm); pr = cm.sum(0); rc = cm.sum(1)
    return np.mean(np.where(pr + rc > 0, 2 * tp / np.maximum(pr + rc, 1), 0))

# ---- load ----
Y, PID, FOLD, P = None, [], [], {m: [] for m in MODELS}
ys = []
for k in FOLDS:
    te = pd.read_csv(SPL / f"fold{k}_test.csv")
    yk = te["label"].map({c: i for i, c in enumerate(CLASSES)}).values
    for m, tag in MODELS.items():
        d = np.load(RES / f"fold{k}_seed{SEED}" / tag / "test_probs.npz")
        assert len(d["y"]) == len(te) and (d["y"] == yk).all(), f"alignment mismatch fold{k} {m}"
        P[m].append(d["p"])
    ys.append(yk); PID += te["patient_id"].tolist(); FOLD += [k] * len(te)
Y = np.concatenate(ys); PID = np.array(PID); FOLD = np.array(FOLD)
P = {m: np.concatenate(v) for m, v in P.items()}
PRED = {m: P[m].argmax(1) for m in MODELS}
print("pooled n =", len(Y), "| patients =", len(set(PID)))

# ---- patient-cluster bootstrap indices (shared across models => paired) ----
upid, inv = np.unique(PID, return_inverse=True)
groups = [np.where(inv == i)[0] for i in range(len(upid))]
boots = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(B)]

def ci(a): return np.percentile(a, [2.5, 97.5])

# ---- per-model summary ----
rows, bs_acc, bs_f1 = [], {}, {}
for m in MODELS:
    pr, pp = PRED[m], P[m]
    fa = [(pr[FOLD == k] == Y[FOLD == k]).mean() for k in FOLDS]
    ff = [macro_f1(Y[FOLD == k], pr[FOLD == k]) for k in FOLDS]
    bs_acc[m] = np.array([(pr[b] == Y[b]).mean() for b in boots])
    bs_f1[m] = np.array([macro_f1(Y[b], pr[b]) for b in boots])
    auc = roc_auc_score(Y, pp, multi_class="ovr", average="macro")
    auc_b = []
    for b in boots[:500]:
        try: auc_b.append(roc_auc_score(Y[b], pp[b], multi_class="ovr", average="macro"))
        except ValueError: pass
    r = dict(model=m,
             acc_fold_mean=np.mean(fa) * 100, acc_fold_sd=np.std(fa, ddof=1) * 100,
             f1_fold_mean=np.mean(ff) * 100, f1_fold_sd=np.std(ff, ddof=1) * 100,
             pooled_acc=(pr == Y).mean() * 100, acc_ci_lo=ci(bs_acc[m])[0] * 100, acc_ci_hi=ci(bs_acc[m])[1] * 100,
             pooled_macroF1=macro_f1(Y, pr) * 100, f1_ci_lo=ci(bs_f1[m])[0] * 100, f1_ci_hi=ci(bs_f1[m])[1] * 100,
             macro_auc=auc, auc_ci_lo=ci(auc_b)[0], auc_ci_hi=ci(auc_b)[1])
    rows.append(r)
summ = pd.DataFrame(rows).round(2)
summ.to_csv(RES / "eval06_summary.csv", index=False)
print(summ.to_string(index=False))

# ---- paired comparisons vs EffNetV2-S baseline ----
prow = []
base = "EffNetV2-S"
for m in MODELS:
    if m == base: continue
    da, df = bs_acc[m] - bs_acc[base], bs_f1[m] - bs_f1[base]
    cm_ok, bm_ok = PRED[m] == Y, PRED[base] == Y
    b01 = int((~bm_ok & cm_ok).sum()); b10 = int((bm_ok & ~cm_ok).sum())
    p_mc = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    fd = np.array([(PRED[m][FOLD == k] == Y[FOLD == k]).mean() - (PRED[base][FOLD == k] == Y[FOLD == k]).mean() for k in FOLDS]) * 100
    prow.append(dict(model=m, vs=base,
        d_acc_pp=(cm_ok.mean() - bm_ok.mean()) * 100, d_acc_ci_lo=ci(da)[0] * 100, d_acc_ci_hi=ci(da)[1] * 100,
        d_f1_pp=(macro_f1(Y, PRED[m]) - macro_f1(Y, PRED[base])) * 100, d_f1_ci_lo=ci(df)[0] * 100, d_f1_ci_hi=ci(df)[1] * 100,
        mcnemar_only_model=b01, mcnemar_only_base=b10, mcnemar_p=p_mc,
        fold_d_mean_pp=fd.mean(), fold_d_sd_pp=fd.std(ddof=1), fold_paired_t_p=ttest_rel(
            [(PRED[m][FOLD == k] == Y[FOLD == k]).mean() for k in FOLDS],
            [(PRED[base][FOLD == k] == Y[FOLD == k]).mean() for k in FOLDS]).pvalue))
pair = pd.DataFrame(prow).round(3)
pair.to_csv(RES / "eval06_paired.csv", index=False)
print(pair.to_string(index=False))

# ---- pooled confusion matrices ----
fig, ax = plt.subplots(2, 4, figsize=(16, 7)); ax = ax.ravel()
cms = {}
for i, m in enumerate(MODELS):
    cm = confusion_matrix(Y, PRED[m], labels=[0, 1, 2]); cms[m] = cm
    ax[i].imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for a in range(3):
        for b in range(3):
            ax[i].text(b, a, f"{cm[a, b]}\n{cm[a, b] / cm[a].sum():.2f}", ha="center", va="center", fontsize=8)
    ax[i].set_xticks(range(3)); ax[i].set_xticklabels(CLASSES, fontsize=8)
    ax[i].set_yticks(range(3)); ax[i].set_yticklabels(CLASSES, fontsize=8); ax[i].set_title(m)
ax[-1].axis("off"); plt.tight_layout(); plt.savefig(RES / "eval06_confusion_pooled.png", dpi=200)
json.dump({m: c.tolist() for m, c in cms.items()}, open(RES / "eval06_confusion_pooled.json", "w"))
print("saved: eval06_summary.csv, eval06_paired.csv, eval06_confusion_pooled.png/json")
