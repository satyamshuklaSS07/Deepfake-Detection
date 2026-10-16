"""
train.py - Dataset banao aur model train karo

1) Video se face frames nikalo:
   python train.py extract --videos data/videos/real --out data/frames/real
   python train.py extract --videos data/videos/fake --out data/frames/fake

2) Train:
   python train.py train --data data/frames --epochs 8

Important: train/validation split VIDEO ke hisaab se hota hai (frame ke hisaab se nahi).
Warna ek hi video ke frames dono mein chale jaate hain aur accuracy jhooti-high dikhti hai.
Frame file naam ka format: <videoname>_<frameno>.jpg (extract yahi banata hai).
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from detector import DEVICE, TFM, FaceTracker, build_net, crop_from_pts


# ------------------------------------------------------------------ extract
def extract(videos_dir, out_dir, every=10, max_per_video=60):
    os.makedirs(out_dir, exist_ok=True)
    paths = [p for p in glob.glob(os.path.join(videos_dir, "*.*"))
             if p.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))]
    for vp in tqdm(paths, desc="videos"):
        name = os.path.splitext(os.path.basename(vp))[0]
        cap = cv2.VideoCapture(vp)
        tracker = FaceTracker()
        i = saved = 0
        while saved < max_per_video:
            ok, frame = cap.read()
            if not ok:
                break
            if i % every == 0:
                pts = tracker.process(frame)
                if pts is not None:
                    face, _ = crop_from_pts(frame, pts)
                    if face.size and min(face.shape[:2]) >= 64:
                        cv2.imwrite(os.path.join(out_dir, f"{name}_{i}.jpg"), face)
                        saved += 1
            i += 1
        cap.release()
        tracker.close()
    print("Done ->", out_dir)


# ------------------------------------------------------------------ train
def video_level_split(dataset, val_frac=0.2, seed=42):
    groups = defaultdict(list)
    for idx, (path, label) in enumerate(dataset.samples):
        vid = os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[0]
        groups[f"{label}/{vid}"].append(idx)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_val = max(1, int(len(keys) * val_frac))
    val_keys = set(keys[:n_val])
    tr = [i for k in keys if k not in val_keys for i in groups[k]]
    va = [i for k in val_keys for i in groups[k]]
    return tr, va


def evaluate(model, loader, fake_idx):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.to(DEVICE)), 1)[:, fake_idx]
            probs += p.cpu().tolist()
            labels += (y == fake_idx).int().tolist()
    probs, labels = np.array(probs), np.array(labels)
    acc = float(((probs > 0.5) == labels).mean())
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(labels, probs > 0.5, labels=[0, 1]).tolist()
    return acc, auc, cm


def train(data, epochs=8, batch=32, lr=1e-4, out="deepfake_model.pt", patience=3):
    train_tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.2),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        TFM,
    ])
    ds_tr = datasets.ImageFolder(data, transform=train_tfm)
    ds_va = datasets.ImageFolder(data, transform=TFM)
    assert set(ds_tr.class_to_idx) == {"real", "fake"}, "data/frames ke andar sirf 'real' aur 'fake' folder hone chahiye"
    fake_idx = ds_tr.class_to_idx["fake"]

    tr_idx, va_idx = video_level_split(ds_tr)
    print(f"Train frames: {len(tr_idx)} | Val frames: {len(va_idx)} | classes: {ds_tr.class_to_idx}")

    tl = DataLoader(Subset(ds_tr, tr_idx), batch_size=batch, shuffle=True,
                    num_workers=2, pin_memory=DEVICE == "cuda")
    vl = DataLoader(Subset(ds_va, va_idx), batch_size=batch, num_workers=2)

    # class imbalance ke liye weights
    counts = np.bincount([ds_tr.samples[i][1] for i in tr_idx], minlength=2).astype(float)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    model = build_net(pretrained=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = DEVICE == "cuda"
    scaler = torch.amp.GradScaler(DEVICE, enabled=use_amp)

    best_auc, bad, metrics = -1.0, 0, {}
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for x, y in tqdm(tl, desc=f"epoch {ep}/{epochs}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE, enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()

        acc, auc, cm = evaluate(model, vl, fake_idx)
        print(f"  loss {run / len(tl):.4f} | val acc {acc:.3f} | val AUC {auc:.3f} | confusion {cm}")
        score = auc if not np.isnan(auc) else acc
        if score > best_auc:
            best_auc, bad = score, 0
            torch.save({"state": model.state_dict(), "classes": ds_tr.class_to_idx, "auc": auc}, out)
            metrics = {"epoch": ep, "val_acc": acc, "val_auc": auc, "confusion_matrix[real,fake]": cm}
            print("  -> best model saved")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Best:", metrics)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("--videos", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--every", type=int, default=10)
    e.add_argument("--max-per-video", type=int, default=60)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--out", default="deepfake_model.pt")

    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.videos, a.out, a.every, a.max_per_video)
    else:
        train(a.data, a.epochs, a.batch, a.lr, a.out)
