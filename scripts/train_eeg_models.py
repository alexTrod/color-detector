#!/usr/bin/env python3
"""Run one of 5 EEG classification models on color epochs; save metrics and confusion matrix."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Import after path
import train_labram as tlm  # noqa: E402

SEED = tlm.SEED
MODELS = ("erp_lr", "erp_svm", "labram_lr", "labram_erp_lr", "eegnet")


def _get_data_and_split(epoch_index_path: Path, label_grouping: str = "basic_4", n_splits: int = 5):
    X, y, groups, channel_names = tlm.load_data(epoch_index_path, label_grouping=label_grouping)
    if len(y) < 2:
        return None, None, None, None, None, None, None
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_groups = len(set(groups.tolist()))
    if n_groups >= 2:
        splitter = GroupShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=SEED)
        splits = list(splitter.split(X, y_enc, groups))
    else:
        from sklearn.model_selection import ShuffleSplit
        splitter = ShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=SEED)
        splits = list(splitter.split(X, y_enc))
    return X, y_enc, groups, channel_names, le, splits, n_groups


def run_erp_lr(X, y_enc, splits, le):
    """ERP summary + Logistic Regression."""
    from sklearn.preprocessing import RobustScaler
    erp = tlm.build_erp_summary_features(X)
    accs, f1s = [], []
    for train_idx, test_idx in splits:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=0.01, random_state=SEED),
        )
        clf.fit(erp[train_idx], y_enc[train_idx])
        pred = clf.predict(erp[test_idx])
        accs.append(float((pred == y_enc[test_idx]).mean()))
        f1s.append(float(f1_score(y_enc[test_idx], pred, average="macro")))
    train_idx, test_idx = splits[0]
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.01, random_state=SEED),
    )
    clf.fit(erp[train_idx], y_enc[train_idx])
    y_pred = clf.predict(erp[test_idx])
    y_true = y_enc[test_idx]
    cm = confusion_matrix(y_true, y_pred).tolist()
    return np.mean(accs), np.mean(f1s), cm, y_true, y_pred


def run_erp_svm(X, y_enc, splits, le):
    """ERP summary + RBF SVM."""
    erp = tlm.build_erp_summary_features(X)
    scaler = StandardScaler()
    erp_scaled = scaler.fit_transform(erp)
    accs, f1s = [], []
    for train_idx, test_idx in splits:
        clf = SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=SEED)
        clf.fit(erp_scaled[train_idx], y_enc[train_idx])
        pred = clf.predict(erp_scaled[test_idx])
        accs.append(float((pred == y_enc[test_idx]).mean()))
        f1s.append(float(f1_score(y_enc[test_idx], pred, average="macro")))
    train_idx, test_idx = splits[0]
    clf = SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=SEED)
    clf.fit(erp_scaled[train_idx], y_enc[train_idx])
    y_pred = clf.predict(erp_scaled[test_idx])
    y_true = y_enc[test_idx]
    cm = confusion_matrix(y_true, y_pred).tolist()
    return np.mean(accs), np.mean(f1s), cm, y_true, y_pred


def run_labram_lr(X, y_enc, splits, le, channel_names, checkpoint_path, device="cpu", batch_size=32):
    """LaBraM features + Logistic Regression."""
    from sklearn.preprocessing import RobustScaler
    labram_feats, _ = tlm.extract_labram_features(X, channel_names, checkpoint_path, device=device, batch_size=batch_size)
    accs, f1s = [], []
    for train_idx, test_idx in splits:
        clf = make_pipeline(
            RobustScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, random_state=SEED),
        )
        clf.fit(labram_feats[train_idx], y_enc[train_idx])
        pred = clf.predict(labram_feats[test_idx])
        accs.append(float((pred == y_enc[test_idx]).mean()))
        f1s.append(float(f1_score(y_enc[test_idx], pred, average="macro")))
    train_idx, test_idx = splits[0]
    clf = make_pipeline(
        RobustScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.1, random_state=SEED),
    )
    clf.fit(labram_feats[train_idx], y_enc[train_idx])
    y_pred = clf.predict(labram_feats[test_idx])
    y_true = y_enc[test_idx]
    cm = confusion_matrix(y_true, y_pred).tolist()
    return np.mean(accs), np.mean(f1s), cm, y_true, y_pred


def run_labram_erp_lr(X, y_enc, splits, le, channel_names, checkpoint_path, device="cpu", batch_size=32):
    """LaBraM + ERP + Logistic Regression."""
    from sklearn.preprocessing import RobustScaler
    labram_feats, _ = tlm.extract_labram_features(X, channel_names, checkpoint_path, device=device, batch_size=batch_size)
    erp = tlm.build_erp_summary_features(X)
    feats = np.concatenate([labram_feats, erp], axis=1)
    accs, f1s = [], []
    for train_idx, test_idx in splits:
        clf = make_pipeline(
            RobustScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=0.005, random_state=SEED),
        )
        clf.fit(feats[train_idx], y_enc[train_idx])
        pred = clf.predict(feats[test_idx])
        accs.append(float((pred == y_enc[test_idx]).mean()))
        f1s.append(float(f1_score(y_enc[test_idx], pred, average="macro")))
    train_idx, test_idx = splits[0]
    clf = make_pipeline(
        RobustScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=0.005, random_state=SEED),
    )
    clf.fit(feats[train_idx], y_enc[train_idx])
    y_pred = clf.predict(feats[test_idx])
    y_true = y_enc[test_idx]
    cm = confusion_matrix(y_true, y_pred).tolist()
    return np.mean(accs), np.mean(f1s), cm, y_true, y_pred


def run_eegnet(X, y_enc, splits, le, device="cpu", epochs=20, lr=1e-3):
    """Lightweight 1D CNN on raw epochs (EEGNet-style)."""
    import torch
    import torch.nn as nn
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # X: (N, C, 1, T) -> (N, C, T)
    x_flat = X[:, :, 0, :].astype(np.float32)
    n_ch, n_t = x_flat.shape[1], x_flat.shape[2]
    n_classes = len(le.classes_)

    class SmallEEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(n_ch, 16, 15, padding=7)
            self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.drop = nn.Dropout(0.25)
            self.fc = nn.Linear(32, n_classes)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = self.drop(x.flatten(1))
            return self.fc(x)

    model = SmallEEGNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    x_t = torch.tensor(x_flat, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_enc, dtype=torch.long, device=device)

    accs, f1s = [], []
    for train_idx, test_idx in splits:
        model = SmallEEGNet().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            model.train()
            perm = np.random.permutation(len(train_idx))
            for i in range(0, len(perm), 32):
                batch = perm[i : i + 32]
                idx = train_idx[batch]
                opt.zero_grad()
                out = model(x_t[idx])
                loss = criterion(out, y_t[idx])
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            out = model(x_t[test_idx])
            pred = out.argmax(dim=1).cpu().numpy()
        y_true = y_enc[test_idx]
        accs.append(float((pred == y_true).mean()))
        f1s.append(float(f1_score(y_true, pred, average="macro")))

    # Refit on first split for confusion matrix
    train_idx, test_idx = splits[0]
    model = SmallEEGNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(len(train_idx))
        for i in range(0, len(perm), 32):
            batch = perm[i : i + 32]
            idx = train_idx[batch]
            opt.zero_grad()
            out = model(x_t[idx])
            loss = criterion(out, y_t[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        y_pred = model(x_t[test_idx]).argmax(dim=1).cpu().numpy()
    y_true = y_enc[test_idx]
    cm = confusion_matrix(y_true, y_pred).tolist()
    return np.mean(accs), np.mean(f1s), cm, y_true, y_pred


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=MODELS)
    parser.add_argument("--epoch-index", type=str, default=str(ROOT / "data/manifests/epoch_index_color.csv"))
    parser.add_argument("--out", type=str, default="", help="Output JSON path (default: runs/eeg_model_comparison/<model>.json)")
    parser.add_argument("--label-grouping", type=str, default="basic_4", choices=("none", "hue_12", "hue_8", "hue_6", "basic_4"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eegnet-epochs", type=int, default=20)
    parser.add_argument("--eegnet-lr", type=float, default=1e-3)
    args = parser.parse_args()

    epoch_index_path = Path(args.epoch_index)
    if not epoch_index_path.exists():
        print(f"Missing {epoch_index_path}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else ROOT / "runs" / "eeg_model_comparison" / f"{args.model}.json"

    result = _get_data_and_split(epoch_index_path, label_grouping=args.label_grouping, n_splits=args.n_splits)
    if result[0] is None:
        print("Not enough samples", file=sys.stderr)
        return 1
    X, y_enc, groups, channel_names, le, splits, _ = result

    checkpoint_path = None
    if args.model in ("labram_lr", "labram_erp_lr"):
        checkpoint_path = tlm.resolve_checkpoint_path("")

    if args.model == "erp_lr":
        acc, f1, cm, y_true, y_pred = run_erp_lr(X, y_enc, splits, le)
    elif args.model == "erp_svm":
        acc, f1, cm, y_true, y_pred = run_erp_svm(X, y_enc, splits, le)
    elif args.model == "labram_lr":
        acc, f1, cm, y_true, y_pred = run_labram_lr(X, y_enc, splits, le, channel_names, checkpoint_path, args.device)
    elif args.model == "labram_erp_lr":
        acc, f1, cm, y_true, y_pred = run_labram_erp_lr(X, y_enc, splits, le, channel_names, checkpoint_path, args.device)
    elif args.model == "eegnet":
        acc, f1, cm, y_true, y_pred = run_eegnet(X, y_enc, splits, le, args.device, args.eegnet_epochs, args.eegnet_lr)
    else:
        return 1

    metrics = {
        "model": args.model,
        "accuracy_mean": float(acc),
        "macro_f1_mean": float(f1),
        "n_splits": args.n_splits,
        "label_classes": list(le.classes_),
        "confusion_matrix": cm,
        "label_grouping": args.label_grouping,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Accuracy: {acc:.4f}  Macro F1: {f1:.4f}")
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
