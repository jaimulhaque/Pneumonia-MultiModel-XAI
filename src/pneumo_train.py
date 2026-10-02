
"""Shared training engine: AdamW + warmup/cosine + class-weighted CE (+LS) + AMP + early stopping."""
import json, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import matplotlib.pyplot as plt

DEFAULT_CFG = dict(epochs=50, lr=1e-4, weight_decay=1e-2, warmup_epochs=2, patience=10,
                   label_smoothing=0.1, amp=True, monitor="val_acc", grad_clip=1.0)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_scheduler(opt, epochs, warmup):
    warm = LinearLR(opt, start_factor=0.01, total_iters=warmup)
    cos = CosineAnnealingLR(opt, T_max=max(1, epochs - warmup))
    return SequentialLR(opt, [warm, cos], milestones=[warmup])


@torch.no_grad()
def predict(model, loader, device, amp=True):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(x)
        ps.append(out.float().softmax(1).cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def metrics_from(y, p, class_weight=None, ls=0.0):
    pred = p.argmax(1)
    nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1)).mean()
    return dict(loss=float(nll), acc=accuracy_score(y, pred),
                bal_acc=balanced_accuracy_score(y, pred),
                macro_f1=f1_score(y, pred, average="macro"))


def train_run(model, train_loader, val_loader, class_weights, run_dir, device, cfg=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    run_dir = Path(run_dir); run_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    crit = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=cfg["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = make_scheduler(opt, cfg["epochs"], cfg["warmup_epochs"])
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    hist, best, best_ep, bad = [], -1.0, -1, 0
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = crit(out.float(), y)
            scaler.scale(loss).backward()
            if cfg["grad_clip"]:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt); scaler.update()
            tl += loss.item() * x.size(0); tn += x.size(0)
            if y.ndim == 1:
                tc += (out.argmax(1) == y).sum().item()
        sched.step()

        yv, pv = predict(model, val_loader, device, use_amp)
        m = metrics_from(yv, pv)
        row = dict(epoch=ep + 1, lr=opt.param_groups[0]["lr"], train_loss=tl / tn,
                   train_acc=(tc / tn) if tc else float("nan"),
                   val_loss=m["loss"], val_acc=m["acc"], val_bal_acc=m["bal_acc"], val_macro_f1=m["macro_f1"])
        hist.append(row)
        score = row[cfg["monitor"]]
        if score > best + 1e-6:
            best, best_ep, bad = score, ep + 1, 0
            torch.save(model.state_dict(), run_dir / "best.pt")
        else:
            bad += 1
        print(f"ep {ep+1:02d} | tr_loss {row['train_loss']:.4f} tr_acc {row['train_acc']:.4f} | "
              f"val_loss {row['val_loss']:.4f} val_acc {row['val_acc']:.4f} f1 {row['val_macro_f1']:.4f} "
              f"| best {best:.4f}@{best_ep} | {time.time()-t0:.0f}s")
        pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
        if bad >= cfg["patience"]:
            print("Early stopping."); break

    h = pd.DataFrame(hist)
    b = h.loc[h["epoch"] == best_ep].iloc[0]
    summary = dict(best_epoch=int(best_ep), best_val_acc=float(b["val_acc"]),
                   best_val_macro_f1=float(b["val_macro_f1"]), best_val_bal_acc=float(b["val_bal_acc"]),
                   epochs_run=int(len(h)), minutes=(time.time() - t0) / 60,
                   params=int(sum(p.numel() for p in model.parameters())), cfg=cfg)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_curves(h, run_dir / "curves.png")
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    return model, summary


def plot_curves(h, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["epoch"], h["train_loss"], label="train"); ax[0].plot(h["epoch"], h["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(h["epoch"], h["train_acc"], label="train"); ax[1].plot(h["epoch"], h["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
