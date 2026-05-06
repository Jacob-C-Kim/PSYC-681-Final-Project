#!/usr/bin/env python3
"""B1 lexical baseline (TF-IDF + Logistic Regression) on real MITweet labels.

Mirrors train_mitweet_real.py's label projection and split policy exactly,
so its predictions CSV is directly paired-bootstrap-comparable against
the transformer's predictions on the same post_ids.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._run_manifest import build_manifest, write_manifest
from scripts.train_mitweet_real import TASKS, load_rows, stratified_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mitweet-csv", type=Path, default=Path("Data/MITweet.csv"))
    ap.add_argument("--max-train", type=int, default=1500)
    ap.add_argument("--max-val", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-predictions-csv", type=Path, default=None)
    ap.add_argument("--model-name", type=str, default="b1_real_mitweet")
    args = ap.parse_args()

    rows = load_rows(args.mitweet_csv)
    train, val = stratified_split(rows, args.seed, args.max_train, args.max_val)
    print(f"[b1] train={len(train)} val={len(val)}")

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, max_df=0.95)
    X_train = tfidf.fit_transform([r.text for r in train])
    X_val = tfidf.transform([r.text for r in val])

    # Train one LR per task, matched class spec.
    rng = np.random.default_rng(args.seed)
    metrics: Dict[str, float] = {}
    preds: Dict[str, np.ndarray] = {}
    probs: Dict[str, np.ndarray] = {}

    truth_attr = {"relevance": "relevance", "economic_direction": "economic", "social_direction": "social"}
    for task, values in TASKS.items():
        attr = truth_attr[task]
        y_train = np.array([getattr(r, attr) for r in train])
        y_val = np.array([getattr(r, attr) for r in val])
        if len(np.unique(y_train)) < 2:
            const = int(np.bincount(y_train).argmax())
            preds[task] = np.full(len(val), const, dtype=int)
            probs[task] = np.zeros((len(val), len(values)))
            probs[task][:, const] = 1.0
            metrics[f"{task}_macro_f1"] = float(f1_score(y_val, preds[task], average="macro", zero_division=0))
            metrics[f"{task}_random_baseline_macro_f1"] = float(f1_score(y_val, rng.choice(np.unique(y_train), size=len(y_val)), average="macro", zero_division=0))
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=1)
        clf.fit(X_train, y_train)
        p = clf.predict(X_val)
        proba = clf.predict_proba(X_val)
        # Reorder probability columns to match the canonical class order in `values`.
        out_probs = np.zeros((len(val), len(values)))
        for col_idx, cls_label in enumerate(clf.classes_):
            if 0 <= int(cls_label) < len(values):
                out_probs[:, int(cls_label)] = proba[:, col_idx]
        preds[task] = p
        probs[task] = out_probs
        metrics[f"{task}_macro_f1"] = float(f1_score(y_val, p, average="macro", zero_division=0))
        metrics[f"{task}_random_baseline_macro_f1"] = float(
            f1_score(y_val, rng.choice(np.unique(y_train), size=len(y_val)), average="macro", zero_division=0)
        )

    metrics["macro_f1_mean"] = float(np.mean([metrics[f"{t}_macro_f1"] for t in TASKS]))
    metrics["random_macro_f1_mean"] = float(
        np.mean([metrics[f"{t}_random_baseline_macro_f1"] for t in TASKS])
    )
    metrics["beats_random"] = bool(metrics["macro_f1_mean"] > metrics["random_macro_f1_mean"])

    print(f"[b1] macro_f1_mean={metrics['macro_f1_mean']:.4f} random={metrics['random_macro_f1_mean']:.4f} beats_random={metrics['beats_random']}")
    for t in TASKS:
        print(f"  {t:22s} F1={metrics[f'{t}_macro_f1']:.4f} (random={metrics[f'{t}_random_baseline_macro_f1']:.4f})")

    payload = {
        "model_name": args.model_name,
        "seed": args.seed,
        "n_train": len(train),
        "n_val": len(val),
        "metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    pred_csv = args.output_predictions_csv or args.output_json.with_name(args.output_json.stem + "_val_predictions.csv")
    pred_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["post_id", "split_name", "model_name"]
    for t in TASKS:
        fieldnames += [f"pred_{t}", f"probs_{t}"]
    with open(pred_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(val):
            row = {"post_id": r.post_id, "split_name": "val", "model_name": args.model_name}
            for t, values in TASKS.items():
                pi = int(preds[t][i])
                row[f"pred_{t}"] = values[pi] if 0 <= pi < len(values) else ""
                row[f"probs_{t}"] = json.dumps({values[j]: float(probs[t][i][j]) for j in range(len(values))}, ensure_ascii=True)
            w.writerow(row)
    print(f"[b1] predictions -> {pred_csv}")

    manifest = build_manifest(
        run_name=f"{args.model_name}_seed{args.seed}",
        seed=args.seed,
        inputs={"mitweet_csv": args.mitweet_csv},
        outputs={"summary_json": str(args.output_json), "val_predictions_csv": str(pred_csv)},
        config={"max_train": args.max_train, "max_val": args.max_val},
        extra={"n_train": len(train), "n_val": len(val), "real_human_labels": True, "real_pretrained_weights": False},
    )
    manifest_path = args.output_json.with_name(args.output_json.stem + "_manifest.json")
    write_manifest(manifest_path, manifest)


if __name__ == "__main__":
    main()
