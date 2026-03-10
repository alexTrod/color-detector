#!/usr/bin/env python3
"""Run 5 EEG models, optionally run 20 autoresearcher iters per model, then summarize."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ROOT / ".venv/bin/python"
TRAIN_EEG = ROOT / "scripts/train_eeg_models.py"
INDEX_COLOR = ROOT / "data/manifests/epoch_index_color.csv"
OUT_DIR = ROOT / "runs/eeg_model_comparison"
LOG_FILE = ROOT / "runs/eeg_model_comparison_summary.log"
MODELS = ("erp_lr", "erp_svm", "labram_lr", "labram_erp_lr", "eegnet")


def log(msg: str) -> None:
    print(msg, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_model(model: str, extra: list[str] | None = None, out_path: Path | None = None) -> dict | None:
    cmd = [str(PY), str(TRAIN_EEG), "--model", model, "--epoch-index", str(INDEX_COLOR)]
    if out_path:
        cmd.extend(["--out", str(out_path)])
    if extra:
        cmd.extend(extra)
    try:
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True, timeout=600)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    p = out_path or (OUT_DIR / f"{model}.json")
    if not p.exists():
        return None
    return json.loads(p.read_text())


def autoresearcher_iters(model: str, n_iters: int, seed: int = 42) -> list[dict]:
    """Run n_iters config variants for one model; save each to _iter<i>.json; return list of metrics."""
    random.seed(seed)
    results = []
    if model == "erp_lr":
        configs = [{"--n-splits": random.choice([5, 10])} for _ in range(n_iters)]
    elif model == "erp_svm":
        configs = [{"--n-splits": random.choice([5, 10])} for _ in range(n_iters)]
    elif model == "labram_lr":
        configs = [{"--n-splits": random.choice([5, 10])} for _ in range(n_iters)]
    elif model == "labram_erp_lr":
        configs = [{"--n-splits": random.choice([5, 10])} for _ in range(n_iters)]
    elif model == "eegnet":
        configs = [
            {
                "--eegnet-epochs": random.choice([15, 20, 25, 30]),
                "--eegnet-lr": random.choice([5e-4, 1e-3, 2e-3]),
            }
            for _ in range(n_iters)
        ]
    else:
        configs = [{}] * n_iters

    for i, cfg in enumerate(configs):
        extra = []
        for k, v in cfg.items():
            extra.extend([k, str(v)])
        out_path = OUT_DIR / f"{model}_iter{i}.json"
        d = run_model(model, extra if extra else None, out_path=out_path)
        if d:
            results.append(d)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-all", action="store_true", help="Run each of the 5 models once")
    parser.add_argument("--autoresearcher", type=int, default=0, metavar="N", help="Run N iters per model (e.g. 20)")
    parser.add_argument("--summarize-only", action="store_true", help="Only build summary from existing JSONs")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"=== {datetime.now().isoformat()} ===")

    if args.summarize_only or not (args.run_all or args.autoresearcher):
        if not args.summarize_only:
            args.run_all = True

    if args.run_all:
        log("Running all 5 models (one run each)...")
        for m in MODELS:
            log(f"  {m} ...")
            run_model(m)
        log("Done run-all.")

    if args.autoresearcher:
        log(f"Autoresearcher: {args.autoresearcher} iters per model...")
        for m in MODELS:
            log(f"  {m} ({args.autoresearcher} iters)...")
            autoresearcher_iters(m, args.autoresearcher)
        log("Done autoresearcher.")

    # Summarize: load all JSONs (including _iter*.json), pick best per model by macro_f1
    log("Building summary...")
    by_model = {}
    for m in MODELS:
        for p in sorted(OUT_DIR.glob(f"{m}.json")) + sorted(OUT_DIR.glob(f"{m}_iter*.json")):
            try:
                r = json.loads(p.read_text())
            except Exception:
                continue
            name = r.get("model", m)
            f1 = r.get("macro_f1_mean") or 0
            if name not in by_model or f1 > (by_model[name].get("macro_f1_mean") or 0):
                by_model[name] = r

    if not by_model:
        log("No result JSONs found.")
        return 0

    rows = []
    for m in MODELS:
        if m in by_model:
            r = by_model[m]
            rows.append({
                "model": m,
                "accuracy": r.get("accuracy_mean", 0),
                "macro_f1": r.get("macro_f1_mean", 0),
                "n_splits": r.get("n_splits", 0),
            })
        else:
            rows.append({"model": m, "accuracy": None, "macro_f1": None, "n_splits": None})

    summary = {
        "results_matrix": rows,
        "confusion_matrices": {r["model"]: r.get("confusion_matrix", []) for r in by_model.values()},
        "label_classes": by_model[list(by_model.keys())[0]].get("label_classes", []) if by_model else [],
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Saved {summary_path}")

    # Results matrix as TSV
    matrix_path = OUT_DIR / "results_matrix.tsv"
    with matrix_path.open("w", encoding="utf-8") as f:
        f.write("model\taccuracy\tmacro_f1\tn_splits\n")
        for row in rows:
            acc = row["accuracy"]
            f1 = row["macro_f1"]
            a_str = f"{acc:.4f}" if acc is not None else ""
            f_str = f"{f1:.4f}" if f1 is not None else ""
            f.write(f"{row['model']}\t{a_str}\t{f_str}\t{row.get('n_splits', '')}\n")
    log(f"Saved {matrix_path}")

    # Write best per model to <model>.json
    for m in MODELS:
        if m in by_model:
            (OUT_DIR / f"{m}.json").write_text(json.dumps(by_model[m], indent=2), encoding="utf-8")

    # Log results matrix as text
    log("Results matrix (best per model):")
    for row in rows:
        acc = row["accuracy"]
        f1 = row["macro_f1"]
        a_str = f"{acc:.4f}" if acc is not None else "N/A"
        f_str = f"{f1:.4f}" if f1 is not None else "N/A"
        log(f"  {row['model']}: accuracy={a_str}  macro_f1={f_str}")

    # Log confusion matrices (best run per model)
    log("Confusion matrices (rows=true, cols=pred):")
    classes = summary.get("label_classes", [])
    for m in MODELS:
        if m not in by_model:
            continue
        cm = by_model[m].get("confusion_matrix", [])
        if not cm:
            continue
        log(f"  [{m}] classes={classes}")
        for row in cm:
            log(f"    {row}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
