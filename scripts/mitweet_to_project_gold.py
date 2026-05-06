#!/usr/bin/env python3
"""Emit project-schema gold_aggregates.csv + raw_posts.csv from MITweet.

Lets the existing scripts/error_analysis.py and scripts/run_seeds.py
"compare" sub-command run unchanged on MITweet predictions: they just
need {post_id, majority_json, soft_distribution_json} for gold and
{post_id, subreddit, topic, text} for raw posts.

Uses the same load_rows / TASKS / projection logic as
scripts/train_mitweet_real.py so the gold side matches the predictions
side exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_mitweet_real import TASKS, load_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mitweet-csv", type=Path, default=Path("Data/MITweet.csv"))
    ap.add_argument("--gold-out", type=Path, default=Path("outputs/real_runs/mitweet_gold_aggregates.csv"))
    ap.add_argument("--raw-out", type=Path, default=Path("outputs/real_runs/mitweet_raw_posts.csv"))
    args = ap.parse_args()

    rows = load_rows(args.mitweet_csv)
    args.gold_out.parent.mkdir(parents=True, exist_ok=True)

    # Map projected labels back to the project's q-key names.
    q_key = {
        "relevance": "q01_relevance",
        "economic_direction": "q07_economic_direction",
        "social_direction": "q08_social_direction",
    }
    # Index -> string label, using the canonical TASKS ordering.
    idx_to_str = {t: TASKS[t] for t in TASKS}
    truth_attr = {"relevance": "relevance", "economic_direction": "economic", "social_direction": "social"}

    with open(args.gold_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["post_id", "majority_json", "soft_distribution_json", "disagreement_entropy", "n_annotators"])
        for r in rows:
            majority = {q_key[t]: idx_to_str[t][getattr(r, truth_attr[t])] for t in TASKS}
            soft = {
                q_key[t]: {idx_to_str[t][getattr(r, truth_attr[t])]: 1.0}
                for t in TASKS
            }
            w.writerow([r.post_id, json.dumps(majority), json.dumps(soft), 0.0, 1])

    with open(args.raw_out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["post_id", "source_dataset", "subreddit", "user_id", "flair",
                    "created_at", "thread_context", "text", "topic", "raw_json"])
        for r in rows:
            w.writerow([r.post_id, "mitweet", r.topic, "", "", "", "", r.text, r.topic, ""])

    print(f"[write] {args.gold_out} ({len(rows)} rows)")
    print(f"[write] {args.raw_out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
