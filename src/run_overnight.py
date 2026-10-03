"""Unattended overnight runner (resumable). Stage A: pilot -> pick variant on VAL. Stage B: fold-0 runs of
selected model + baselines (same config), then TEST metrics for a first look. Safe to stop/restart."""
import argparse, json, sys, time, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, cohen_kappa_score, precision_recall_fscore_support)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pneumo_data import make_loaders, CLASSES
from pneumo_models import build_model, count_params
from pneumo_train import set_seed, train_run, predict

ap = argparse.ArgumentParser()
ap.add_argument("--project-root", type=Path, default=HERE.parent)
ap.add_argument("--cache-dir", type=Path, default=Path.home() / "cache_384")
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--pilot-epochs", type=int, default=15)
ap.add_argument("--pilot-patience", type=int, default=5)
ap.add_argument("--main-epochs", type=int, default=30)
ap.add_argument("--main-patience", type=int, default=8)
ap.add_argument("--budget-hours", type=float, default=4.75, help="do not START a new run after this")
ap.add_argument("--no-pretrained", action="store_true")
args = ap.parse_args()

ROOT = args.project_root
SPLITS = ROOT / "splits"
PILOT_DIR = ROOT / "results" / "pilot"
MAIN_DIR = ROOT / "results" / f"fold{args.fold}_seed{args.seed}"
MAIN_DIR.mkdir(parents=True, exist_ok=True); PILOT_DIR.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRE = not args.no_pretrained
T0 = time.time()
print(f"Device: {device} | cache: {args.cache_dir} | budget {args.budget_hours} h", flush=True)


def elapsed_h():
    return (time.time() - T0) / 3600


def loaders():
    set_seed(args.seed)
    return make_loaders(args.fold, SPLITS, args.cache_dir, batch_size=args.batch,
                        num_workers=args.workers, seed=args.seed)


def run_one(spec, run_dir, epochs, patience):
    sf = run_dir / "summary.json"
    if sf.exists():
        print("Skip (done):", spec, flush=True)
        return json.loads(sf.read_text()), None
    print(f"\n===== {spec} | {epochs} ep / patience {patience} | elapsed {elapsed_h():.2f} h =====", flush=True)
    tr, va, te, cw, _ = loaders()
    model = build_model(spec, pretrained=PRE)
    model, s = train_run(model, tr, va, cw, run_dir, device, dict(epochs=epochs, patience=patience))
    return s, (model, te)


# ---------------- Stage A: pilot (validation only) ----------------
CANDS = [f"proposed:{b}:{a}" for b in ("convnext_tiny", "efficientnet_v2_s") for a in ("eca", "cbam")]
rows = []
for spec in CANDS:
    run_dir = PILOT_DIR / spec.replace(":", "_")
    if not (run_dir / "summary.json").exists() and elapsed_h() > args.budget_hours:
        print("Budget reached, skip pilot:", spec); continue
    try:
        s, h = run_one(spec, run_dir, args.pilot_epochs, args.pilot_patience)
        rows.append({"model": spec, "best_val_acc": s["best_val_acc"], "best_val_macro_f1": s["best_val_macro_f1"],
                     "best_epoch": s["best_epoch"], "params_M": s["params"] / 1e6, "minutes": s["minutes"]})
        if h: del h
    except Exception:
        print("FAILED:", spec); traceback.print_exc()
    torch.cuda.empty_cache()

res = pd.DataFrame(rows).sort_values("best_val_acc", ascending=False).reset_index(drop=True)
res.to_csv(PILOT_DIR / "pilot_results.csv", index=False)
print("\nPILOT RESULTS (validation):\n", res.to_string(), flush=True)

top = res["best_val_acc"].max()
sel = res[res["best_val_acc"] >= top - 0.005].sort_values("params_M").iloc[0]
_, backbone, attn = sel["model"].split(":")
selected = {"spec": sel["model"], "backbone": backbone, "attn": attn,
            "val_acc": float(sel["best_val_acc"]), "params_M": float(sel["params_M"])}
(ROOT / "selected_model.json").write_text(json.dumps(selected, indent=2))
print("SELECTED:", selected, flush=True)

# ---------------- Stage B: same config for selected + baselines ----------------
order = [selected["spec"], f"baseline:{backbone}", "baseline:efficientnet_b2", "baseline:resnet50",
         "baseline:convnext_tiny", "baseline:swin_t"]
specs = list(dict.fromkeys(order))


def test_metrics(model, te, run_dir):
    y, p = predict(model, te, device, True)
    pred = p.argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    m = dict(test_acc=accuracy_score(y, pred), test_bal_acc=balanced_accuracy_score(y, pred),
             test_macro_f1=f1_score(y, pred, average="macro"), test_mcc=matthews_corrcoef(y, pred),
             test_kappa=cohen_kappa_score(y, pred))
    for i, c in enumerate(CLASSES):
        m[f"{c}_P"], m[f"{c}_R"], m[f"{c}_F1"] = float(pr[i]), float(rc[i]), float(f1[i])
    (run_dir / "test_metrics.json").write_text(json.dumps(m, indent=2))
    np.savez(run_dir / "test_probs.npz", y=y, p=p)
    return m


out = []
for spec in specs:
    run_dir = MAIN_DIR / spec.replace(":", "_")
    if not (run_dir / "summary.json").exists() and elapsed_h() > args.budget_hours:
        print("Budget reached, skip:", spec); continue
    try:
        s, h = run_one(spec, run_dir, args.main_epochs, args.main_patience)
        if h is not None:
            model, te = h
            test_metrics(model, te, run_dir); del model
        tm = json.loads((run_dir / "test_metrics.json").read_text()) if (run_dir / "test_metrics.json").exists() else {}
        out.append({"model": spec, "params_M": s["params"] / 1e6, "best_epoch": s["best_epoch"],
                    "val_acc": s["best_val_acc"], "minutes": s["minutes"], **tm})
        pd.DataFrame(out).to_csv(MAIN_DIR / "overnight_summary.csv", index=False)
    except Exception:
        print("FAILED:", spec); traceback.print_exc()
    torch.cuda.empty_cache()

print("\nFOLD-%d RESULTS (preliminary):\n" % args.fold, pd.DataFrame(out).to_string(), flush=True)
print(f"\nDONE in {elapsed_h():.2f} h", flush=True)
