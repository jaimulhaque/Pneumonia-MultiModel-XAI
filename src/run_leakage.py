"""Leakage experiment: same model/config, image-level (leaky) vs patient-level split. Resumable.
Image-level splits -> splits_imglevel/ (same file layout, so make_loaders is used unchanged).
Runs -> results/leakage_img/fold{k}_seed{s}/<tag>/. Patient-level reference = results/cv_runs.csv."""
import argparse, gc, json, sys, time, traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
                             matthews_corrcoef, precision_recall_fscore_support)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pneumo_data import CLASSES, make_loaders
from pneumo_models import build_model
from pneumo_train import predict, set_seed, train_run

ap = argparse.ArgumentParser()
ap.add_argument("--project-root", type=Path, default=HERE.parent)
ap.add_argument("--cache-dir", type=Path, default=Path.home() / "cache_384")
ap.add_argument("--models", nargs="+", default=["baseline:efficientnet_v2_s"])
ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--patience", type=int, default=8)
ap.add_argument("--budget-hours", type=float, default=4.0, help="do not START a new run after this")
ap.add_argument("--aggregate-only", action="store_true")
args = ap.parse_args()

ROOT = args.project_root
SPLITS, IMG_SPLITS = ROOT / "splits", ROOT / "splits_imglevel"
RES = ROOT / "results"
LEAK = RES / "leakage_img"
LEAK.mkdir(parents=True, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T0 = time.time()


def elapsed_h():
    return (time.time() - T0) / 3600


# ---------------- image-level splits (ignore patient_id) ----------------
def make_image_level_splits():
    IMG_SPLITS.mkdir(exist_ok=True)
    if not all((IMG_SPLITS / f"fold{k}_{r}.csv").exists() for k in range(5) for r in ("train", "val", "test")):
        df = pd.concat([pd.read_csv(SPLITS / f"fold{k}_test.csv") for k in range(5)], ignore_index=True)
        assert df["image_path"].is_unique, "duplicate image_path across test folds"
        fid = np.full(len(df), -1)
        skf = StratifiedKFold(5, shuffle=True, random_state=args.seed)
        for k, (_, te_idx) in enumerate(skf.split(df, df["label"])):
            fid[te_idx] = k
        assert (fid >= 0).all()
        for k in range(5):
            df[fid == k].to_csv(IMG_SPLITS / f"fold{k}_test.csv", index=False)
            df[fid == (k + 1) % 5].to_csv(IMG_SPLITS / f"fold{k}_val.csv", index=False)
            df[(fid != k) & (fid != (k + 1) % 5)].to_csv(IMG_SPLITS / f"fold{k}_train.csv", index=False)
        print(f"Created image-level splits in {IMG_SPLITS} (n={len(df)})", flush=True)
    rows = []
    for k in range(5):
        tr = pd.read_csv(IMG_SPLITS / f"fold{k}_train.csv")
        va = pd.read_csv(IMG_SPLITS / f"fold{k}_val.csv")
        te = pd.read_csv(IMG_SPLITS / f"fold{k}_test.csv")
        trp = set(tr["patient_id"]); allp = trp | set(va["patient_id"])
        rows.append(dict(fold=k, n_train=len(tr), n_val=len(va), n_test=len(te),
                         test_imgs_patient_in_train_pct=100 * te["patient_id"].isin(trp).mean(),
                         test_patients_in_train_pct=100 * pd.Series(te["patient_id"].unique()).isin(trp).mean(),
                         val_imgs_patient_in_train_pct=100 * va["patient_id"].isin(trp).mean(),
                         label_dist_test=te["label"].value_counts().to_dict().__str__()))
    st = pd.DataFrame(rows)
    st.to_csv(RES / "leakage_split_stats.csv", index=False)
    print("\nIMAGE-LEVEL SPLIT LEAK STATS (patient-level split = 0% by construction):\n",
          st.drop(columns="label_dist_test").round(2).to_string(index=False), flush=True)
    return st


def test_metrics(model, te, run_dir):
    model.eval()
    with torch.no_grad():
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


def aggregate(stats):
    rows = []
    for spec in args.models:
        tag = spec.replace(":", "_")
        for k in range(5):
            rd = LEAK / f"fold{k}_seed{args.seed}" / tag
            if (rd / "summary.json").exists() and (rd / "test_metrics.json").exists():
                sm = json.loads((rd / "summary.json").read_text()); tm = json.loads((rd / "test_metrics.json").read_text())
                rows.append(dict(model=spec, fold=k, best_epoch=sm["best_epoch"], val_acc=sm["best_val_acc"], **tm))
    if not rows:
        print("No finished leakage runs yet."); return
    img = pd.DataFrame(rows); img.to_csv(RES / "leakage_img_runs.csv", index=False)
    pat = pd.read_csv(RES / "cv_runs.csv"); pat = pat[pat["seed"] == args.seed]
    out, perfold = [], []
    for spec in args.models:
        a = img[img["model"] == spec].set_index("fold"); b = pat[pat["model"] == spec].set_index("fold")
        common = sorted(set(a.index) & set(b.index))
        if len(common) < 2: continue
        for met in ["test_acc", "test_macro_f1", "test_mcc", "VIRUS_F1", "val_acc"]:
            x, y = a.loc[common, met].values * 100, b.loc[common, met].values * 100
            out.append(dict(model=spec, metric=met, n_folds=len(common),
                            image_level_mean=x.mean(), image_level_sd=x.std(ddof=1),
                            patient_level_mean=y.mean(), patient_level_sd=y.std(ddof=1),
                            inflation_pp=(x - y).mean(), inflation_sd=(x - y).std(ddof=1),
                            paired_t_p=ttest_rel(x, y).pvalue))
        for k in common:
            perfold.append(dict(model=spec, fold=k, image_acc=a.loc[k, "test_acc"] * 100, patient_acc=b.loc[k, "test_acc"] * 100,
                                diff_pp=(a.loc[k, "test_acc"] - b.loc[k, "test_acc"]) * 100,
                                leak_img_pct=stats.loc[stats["fold"] == k, "test_imgs_patient_in_train_pct"].iloc[0]))
    o = pd.DataFrame(out).round(3); pf = pd.DataFrame(perfold).round(2)
    o.to_csv(RES / "leakage_summary.csv", index=False); pf.to_csv(RES / "leakage_perfold.csv", index=False)
    print("\nPER-FOLD (test acc %):\n", pf.to_string(index=False))
    print("\nLEAKAGE SUMMARY (image-level vs patient-level, %):\n", o.to_string(index=False), flush=True)


stats = make_image_level_splits()
if not args.aggregate_only:
    print(f"Device {device} | models {args.models} | folds {args.folds} | budget {args.budget_hours} h", flush=True)
    for fold in args.folds:
        main_dir = LEAK / f"fold{fold}_seed{args.seed}"; main_dir.mkdir(parents=True, exist_ok=True)
        for spec in args.models:
            run_dir = main_dir / spec.replace(":", "_")
            if (run_dir / "summary.json").exists() and (run_dir / "test_metrics.json").exists():
                print(f"Skip (done): fold{fold} {spec}", flush=True); continue
            if elapsed_h() > args.budget_hours:
                print(f"Budget reached, skip: fold{fold} {spec}", flush=True); continue
            print(f"\n===== IMG-LEVEL fold{fold} {spec} | elapsed {elapsed_h():.2f} h =====", flush=True)
            model = None
            try:
                set_seed(args.seed)
                tr, va, te, cw, _ = make_loaders(fold, IMG_SPLITS, args.cache_dir, batch_size=args.batch,
                                                 num_workers=args.workers, seed=args.seed)
                model = build_model(spec, pretrained=True)
                if (run_dir / "summary.json").exists():
                    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu")); model.to(device)
                else:
                    model, _ = train_run(model, tr, va, cw, run_dir, device, dict(epochs=args.epochs, patience=args.patience))
                test_metrics(model, te, run_dir)
                aggregate(stats)
            except Exception:
                print("FAILED:", spec); traceback.print_exc()
            finally:
                if model is not None: del model
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
aggregate(stats)
print(f"\nDONE in {elapsed_h():.2f} h", flush=True)
