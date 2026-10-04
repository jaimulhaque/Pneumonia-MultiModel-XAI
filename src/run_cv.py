"""Resumable k-fold / multi-seed runner. Same config & result layout as run_overnight.py
(results/fold{k}_seed{s}/<model_tag>/), finished runs are skipped. Fold-major order: finished folds are complete."""
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
from pneumo_models import build_model
from pneumo_train import set_seed, train_run, predict

ap = argparse.ArgumentParser()
ap.add_argument("--project-root", type=Path, default=HERE.parent)
ap.add_argument("--cache-dir", type=Path, default=Path.home() / "cache_384")
ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--seeds", type=int, nargs="+", default=[42])
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--patience", type=int, default=8)
ap.add_argument("--budget-hours", type=float, default=8.0, help="do not START a new run after this")
ap.add_argument("--no-pretrained", action="store_true")
args = ap.parse_args()

ROOT = args.project_root
SPLITS = ROOT / "splits"
RES = ROOT / "results"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T0 = time.time()
MODELS = ["baseline:efficientnet_v2_s", "proposed:efficientnet_v2_s:eca", "baseline:efficientnet_b2",
          "baseline:resnet50", "proposed:efficientnet_v2_s:cbam", "baseline:convnext_tiny", "baseline:swin_t"]
TAG2SPEC = {m.replace(":", "_"): m for m in MODELS}
print(f"Device {device} | folds {args.folds} | seeds {args.seeds} | budget {args.budget_hours} h", flush=True)


def elapsed_h():
    return (time.time() - T0) / 3600


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


def aggregate():
    rows = []
    for d in sorted(RES.glob("fold*_seed*")):
        k, s = d.name.replace("fold", "").split("_seed")
        for rd in sorted(p for p in d.iterdir() if p.is_dir()):
            if (rd / "summary.json").exists() and (rd / "test_metrics.json").exists():
                sm = json.loads((rd / "summary.json").read_text())
                tm = json.loads((rd / "test_metrics.json").read_text())
                rows.append({"model": TAG2SPEC.get(rd.name, rd.name),
                             "fold": int(k), "seed": int(s), "params_M": sm["params"] / 1e6,
                             "best_epoch": sm["best_epoch"], "val_acc": sm["best_val_acc"],
                             "minutes": sm["minutes"], **tm})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(RES / "cv_runs.csv", index=False)
    cols = ["test_acc", "test_bal_acc", "test_macro_f1", "test_mcc", "test_kappa", "VIRUS_F1"]
    g = df.groupby("model")[cols]
    summ = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd")).join(df.groupby("model").size().rename("n_runs"))
    summ.sort_values("test_acc_mean", ascending=False).to_csv(RES / "cv_summary.csv")
    print("\nCV SUMMARY so far:\n", summ.sort_values("test_acc_mean", ascending=False)[
        ["test_acc_mean", "test_acc_sd", "test_macro_f1_mean", "test_macro_f1_sd", "n_runs"]].round(4).to_string(), flush=True)


for seed in args.seeds:
    for fold in args.folds:
        main_dir = RES / f"fold{fold}_seed{seed}"
        main_dir.mkdir(parents=True, exist_ok=True)
        for spec in MODELS:
            run_dir = main_dir / spec.replace(":", "_")
            if (run_dir / "summary.json").exists() and (run_dir / "test_metrics.json").exists():
                print(f"Skip (done): fold{fold} seed{seed} {spec}", flush=True); continue
            if elapsed_h() > args.budget_hours:
                print(f"Budget reached, skip: fold{fold} seed{seed} {spec}", flush=True); continue
            print(f"\n===== fold{fold} seed{seed} {spec} | elapsed {elapsed_h():.2f} h =====", flush=True)
            try:
                set_seed(seed)
                tr, va, te, cw, _ = make_loaders(fold, SPLITS, args.cache_dir, batch_size=args.batch,
                                                 num_workers=args.workers, seed=seed)
                model = build_model(spec, pretrained=not args.no_pretrained)
                if (run_dir / "summary.json").exists():          # trained earlier, only test metrics missing
                    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
                    model.to(device)
                else:
                    model, _ = train_run(model, tr, va, cw, run_dir, device,
                                         dict(epochs=args.epochs, patience=args.patience))
                test_metrics(model, te, run_dir)
                del model
                aggregate()
            except Exception:
                print("FAILED:", spec); traceback.print_exc()
            torch.cuda.empty_cache()

aggregate()
print(f"\nDONE in {elapsed_h():.2f} h", flush=True)
