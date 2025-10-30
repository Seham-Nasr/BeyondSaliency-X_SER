#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a CNN (ResNet-like) for Speech Emotion Recognition on precomputed spectrograms.

Assumptions:
- A pickled DataFrame at --df_path where column 'Emotions' holds labels (str or int),
  and columns 1..n contain tuples (file_path, spectrogram_array).
- Each spectrogram is 2D (freq x time). We add a channel dim in the Dataset.

Example:
  python3 Models/trainSER.py \
    --df_path /homes/snasr/2026/speech26/data/TESS_df.pkl \
    --batch_size 32 --epochs 50 --lr 3e-4 \
    --checkpoint checkpoint/best_model_tess.pth \
    --include_emotions anger disgust fear happy pleasant_surprise sad neutral
 python3 Models/trainSER.py     --df_path /homes/snasr/2026/speech26/data/TESS_df.pkl     --batch_size 32 --epochs 50 --lr 3e-4     --checkpoint Models/checkpoint/best_model_tess.pth 
python3 Models/trainSER.py     --df_path /homes/snasr/2026/speech26/data/Crema-D.pkl     --batch_size 32 --epochs 100 --lr 3e-4     --checkpoint Models/checkpoint/best_model_Crema-D.pth  --include_emotions fear neutral happy angry disgust surprise sad

"""

import os
import sys
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Reproducibility & device
# ---------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def get_device(pref: Optional[str] = None) -> torch.device:
    if pref is not None:
        pref = pref.lower()
        if pref == "cpu":
            return torch.device("cpu")
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Dataset
# ---------------------------
class SpectrogramDataset(Dataset):
    """
    Expects X as array/list where each element is a tuple: (file_path, spectrogram_array)
    and y as integer labels.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        _, spec = self.X[idx]
        label = self.y[idx]
        # (1, F, T)
        spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return spec, label

# ---------------------------
# Model
# ---------------------------
class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
            if in_ch != out_ch or stride != 1 else None
        )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out = F.relu(out + identity)
        return out

class ResNetSmall(nn.Module):
    def __init__(self, n_classes:int):
        super(ResNetSmall, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.layer1 = ResNetBlock(32, 64, stride=1)
        self.layer2 = ResNetBlock(64, 128, stride=2)

        # Increased dropout rate
        self.dropout = nn.Dropout(0.50)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        x = self.layer1(x)
        x = self.layer2(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

# ---------------------------
# Data helpers
# ---------------------------
def load_dataframe(df_path: str) -> pd.DataFrame:
    df = pd.read_pickle(df_path)
    if "Emotions" not in df.columns:
        raise ValueError("DataFrame must contain an 'Emotions' column.")
    return df

def filter_by_emotions(df: pd.DataFrame, include: Optional[List[str]]) -> pd.DataFrame:
    if not include:
        return df.reset_index(drop=True)
    emo = df["Emotions"]
    if emo.dtype == object:
        return df[df["Emotions"].isin(include)].reset_index(drop=True)
    # try coerce include to ints if labels are numeric
    try:
        include_int = [int(e) for e in include]
        return df[df["Emotions"].isin(include_int)].reset_index(drop=True)
    except Exception:
        return df.reset_index(drop=True)

def remap_labels_contiguously(df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
    """Map labels to 0..C-1 (always), return df with int labels and the ordered class names."""
    uniq = sorted(pd.unique(df["Emotions"]))
    label_to_idx = {old: i for i, old in enumerate(uniq)}
    df = df.copy()
    df["Emotions"] = df["Emotions"].map(label_to_idx).astype(int)
    return df, uniq

def split_df(df: pd.DataFrame, test_size=0.1, val_size=0.1, seed=42):
    X = df.iloc[:, 1:].values  # assumes col 0 is meta; columns 1.. contain tuples
    y = df["Emotions"].values
    from sklearn.model_selection import train_test_split
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_ratio, random_state=seed, stratify=y_tmp)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# ---------------------------
# Training / evaluation
# ---------------------------
def accuracy_from_logits(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def run_epoch(model, loader, optimizer, scaler, device, train=True, autocast_ctx=None):
    model.train(mode=train)
    total_loss, total_acc, n = 0.0, 0.0, 0
    criterion = nn.CrossEntropyLoss()
    n_classes = model.fc.out_features if hasattr(model, "fc") else None

    for specs, labels in loader:
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if n_classes is not None:
            # Guard against out-of-range targets
            min_t = int(labels.min().item())
            max_t = int(labels.max().item())
            assert 0 <= min_t and max_t < n_classes, \
                f"Label out of range: [{min_t}, {max_t}] with n_classes={n_classes}"

        with torch.set_grad_enabled(train):
            with (autocast_ctx() if autocast_ctx is not None else torch.no_grad()):
                logits = model(specs)
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * specs.size(0)
        total_acc += accuracy_from_logits(logits, labels) * specs.size(0)
        n += specs.size(0)

    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    criterion = nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0

    for specs, labels in loader:
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(specs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * specs.size(0)
        n += specs.size(0)

        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds) if len(all_preds) else np.array([])
    all_targets = np.concatenate(all_targets) if len(all_targets) else np.array([])
    avg_loss = total_loss / max(n, 1)
    acc = float((all_preds == all_targets).mean()) if n > 0 else 0.0
    return avg_loss, acc, all_preds, all_targets

def save_checkpoint(model, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)

# ---------------------------
# CLI / main
# ---------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train SER model on selected data (spectrograms).")
    parser.add_argument("--df_path", type=str, required=True, help="Pickled DataFrame path.")
    parser.add_argument("--include_emotions", type=str, nargs="*", default=None,
                        help="Subset of emotions to include (strings or ints).")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default="checkpoint/best_model.pth")
    parser.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'. Defaults to auto.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop_patience", type=int, default=7)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)
    use_cuda = (device.type == "cuda")

    # AMP setup (new torch.amp API; only when CUDA is available)
    if use_cuda:
        scaler = torch.amp.GradScaler('cuda')
        def autocast_ctx():
            return torch.amp.autocast('cuda')
    else:
        scaler = None
        from contextlib import nullcontext
        def autocast_ctx():
            return nullcontext()

    # Load & optionally filter
    df = load_dataframe(args.df_path)
    df = filter_by_emotions(df, args.include_emotions)
    df, class_names = remap_labels_contiguously(df)  # ensures targets in [0..C-1]
    n_classes = len(class_names)

    # Split
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_df(df, test_size=0.1, val_size=0.1, seed=args.seed)

    # Datasets / Loaders
    train_ds = SpectrogramDataset(X_tr, y_tr)
    val_ds   = SpectrogramDataset(X_va, y_va)
    test_ds  = SpectrogramDataset(X_te, y_te)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=use_cuda)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=use_cuda)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=use_cuda)

    # Model / Optimizer
    model = ResNetSmall(n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train loop with early stopping on val accuracy
    best_val_acc = -1.0
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, scaler, device, train=True, autocast_ctx=autocast_ctx)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, device)

        print(f"Epoch {epoch:03d} | train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val_loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            epochs_no_improve = 0
            save_checkpoint(model, args.checkpoint)
            print(f"  ↳ Saved best to {args.checkpoint}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print("Early stopping triggered.")
                break

    # Load best and test
    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    te_loss, te_acc, preds, targets = evaluate(model, test_loader, device)
    print(f"\nTest  | loss {te_loss:.4f} acc {te_acc:.4f}")

    # Reports
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        print("\nClassification report:")
        print(classification_report(targets, preds, digits=4))
        print("Confusion matrix:")
        print(confusion_matrix(targets, preds))
    except Exception as e:
        print(f"(sklearn metrics unavailable or failed: {e})")

    # Persist label map for inference
    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    label_map_path = os.path.splitext(args.checkpoint)[0] + "_labels.npy"
    np.save(label_map_path, np.array(class_names, dtype=object))
    print(f"Saved label map: {label_map_path}")

if __name__ == "__main__":
    main()
