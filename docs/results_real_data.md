# Real-Data Results: B1 vs B2 on MITweet (3 seeds)

This document records the first real-data run of the project's
infrastructure. It is intentionally narrow in scope: it uses real
human-annotated MITweet labels and a real pretrained RoBERTa-base
encoder, but it does not include the Politosphere weak-supervision
stage that the original B3-B6 ladder requires. The reason is recorded
in the run manifests as
`weak_pretraining_dataset = blocked_zenodo_unreachable`: the experiment
sandbox cannot reach Zenodo, so the Politosphere `comments_*.bz2` files
cannot be downloaded here. The Politosphere half of the experiment
must be rerun in an environment with full network access.

## What this run does include

- Pretrained RoBERTa-base (124.6 M parameters), loaded from the legacy
  `s3.amazonaws.com/models.huggingface.co/bert/` mirror because the
  HuggingFace Hub itself returns 403 in this sandbox. Files cached
  under `models/.cache/roberta-base/` (gitignored).
- Real human-annotated MITweet labels (12,594 tweets across 14
  political topics).
- The same label projection on both the B1 (TF-IDF + LR) and
  B2 (RoBERTa fine-tune) sides, so per-row predictions pair across
  models and a paired bootstrap is meaningful.
- Three seeds (42, 43, 44), stratified train/val split by topic
  capped at 1500 train / 400 val per seed.
- Standard reproducibility manifest per run with seed, data hashes,
  git commit, environment versions.
- Per-row predictions (`pred_<task>`, `probs_<task>`) for every model
  on every seed, fed into the existing `scripts/error_analysis.py`
  and `scripts/run_seeds.py compare`.

## What this run does NOT include

- B3 / B4 / B5 (weak-pretrained transformers) -- they require
  Politosphere weak labels.
- B6 (hierarchical) -- needs the same weak labels.
- Leave-one-topic-out and leave-one-community-out evaluation
  regimes -- the project's split conventions assume Reddit-style
  community + topic metadata. MITweet has topic but not subreddit.
- The MITweet projection is a deliberate simplification of the full
  facet structure: relevance comes from any active R1-R5; economic
  direction is the majority sign of I1-I4; social direction is the
  majority sign of I5-I12; rows where every relevant facet is N/A
  fall to the neutral class.

## Aggregate metrics (3-seed mean +/- std, 95% CI)

| Task                | B1 (TF-IDF + LR)            | B2 (RoBERTa-base fine-tune) | Random            |
|---------------------|-----------------------------|-----------------------------|-------------------|
| relevance           | 0.708 +/- 0.077 [0.62,0.80] | 0.743 +/- 0.064 [0.67,0.82] | ~0.41             |
| economic_direction  | 0.459 +/- 0.021 [0.44,0.48] | 0.341 +/- 0.017 [0.32,0.36] | ~0.21             |
| social_direction    | 0.534 +/- 0.015 [0.52,0.55] | 0.579 +/- 0.022 [0.55,0.60] | ~0.34             |
| **mean macro-F1**   | **0.567 +/- 0.033**         | **0.554 +/- 0.024**         | **0.314**         |

Both models comfortably beat random (Checkpoint B). Their overall
mean macro-F1 is within seed variance; the per-task pattern matters.

## Paired bootstrap (B1 vs B2, 400 shared val posts per seed, 2000 resamples)

| Task                | seed 42                                    | seed 43                                     | seed 44                                    |
|---------------------|--------------------------------------------|---------------------------------------------|--------------------------------------------|
| relevance           | tie  B-A=+0.023  CI=[-0.041,+0.086]  p=0.47 | tie  B-A=+0.029  CI=[-0.066,+0.119]  p=0.55  | tie  B-A=+0.051  CI=[-0.060,+0.162]  p=0.36 |
| economic_direction  | **A>B**  B-A=-0.129  CI=[-0.230,-0.037]  p=0.008 | **A>B**  B-A=-0.096  CI=[-0.165,-0.022]  p=0.014 | **A>B**  B-A=-0.131  CI=[-0.189,-0.060]  p<0.001 |
| social_direction    | tie  B-A=+0.019  CI=[-0.030,+0.069]  p=0.48 | **B>A**  B-A=+0.086  CI=[+0.037,+0.132]  p<0.001 | tie  B-A=+0.030  CI=[-0.020,+0.082]  p=0.26 |

Reading: **B-A** is `B2_macro_F1 - B1_macro_F1` on the same val rows.
A 95% CI that excludes zero is treated as a directional verdict.

The cross-seed pattern is consistent: B1 significantly beats B2 on
**economic_direction at every seed** (p < 0.02). B2 wins
**social_direction** at one seed, ties at the others. Relevance is
always a tie within bootstrap variance.

## Per-class F1 and class support (B2, 3-seed mean)

| Task                | Class | Support mean | F1 mean +/- std |
|---------------------|-------|--------------|-----------------|
| relevance           |   0   |  32          | 0.514 +/- 0.127 |
| relevance           |   1   | 368          | 0.971 +/- 0.002 |
| economic_direction  |  -1   |  24          | 0.067 +/- 0.060 |
| economic_direction  |   0   | 365          | 0.955 +/- 0.010 |
| economic_direction  |  +1   |  11          | 0.000 +/- 0.000 |
| social_direction    |  -1   | 192          | 0.717 +/- 0.023 |
| social_direction    |   0   | 121          | 0.625 +/- 0.045 |
| social_direction    |  +1   |  87          | 0.394 +/- 0.085 |

The reason B1 outperforms B2 on `economic_direction` is visible here:
91% of economic gold labels are class 0 (neutral). B1 fits with
`class_weight="balanced"`, which lifts minority-class recall; B2
fine-tunes with vanilla cross-entropy and collapses onto the majority
class (F1 = 0.07 / 0.00 / 0.96 across the three classes). On the more
balanced `social_direction`, the transformer's representations help
and B2 catches up (and at seed 43 surpasses B1 by 0.09 macro-F1).

## Confidence calibration: correct vs error (B2, 3-seed mean)

| Task                | mean conf when correct | mean conf when wrong | gap   |
|---------------------|------------------------|----------------------|-------|
| relevance           | 0.957                  | 0.849                | +0.107 |
| economic_direction  | 0.959                  | 0.791                | +0.168 |
| social_direction    | 0.686                  | 0.577                | +0.109 |

There is a real gap between correct-prediction confidence and
error-prediction confidence at every task -- meaning abstention has
something to work with. The gap is largest on `economic_direction`
(+0.17), which is also the task where temperature-scaling +
abstention would help the most. A high-confidence-error rate at
threshold 0.7 of 13% on `social_direction` means roughly one in eight
predictions the model commits to with conf >= 0.7 are still wrong.

## Single most actionable next step

These results say what the next iteration should change before
relying on the transformer ladder:

1. **Add class-weighted (or focal) loss to all ideology heads.** B2's
   collapse on economic_direction is a class-balance bug, not an
   architectural bug.
2. **Train longer than 2 epochs once class weighting is in place.**
   The CPU budget here capped training at 2 epochs / 1500 examples;
   on a GPU this should be dialed up considerably.
3. **Run the same scripts on Politosphere weak labels** (B3 / B4 /
   B5 / B6) once Zenodo is reachable. The infrastructure -- weak-label
   builder, manifests, predictions schema, error analysis, paired
   bootstrap -- is already in place.

## Reproducibility

Every run wrote a `run_manifest.json` next to its summary. Each
manifest carries: SHA-256 of the input MITweet CSV (so the gold side
is verifiable), seed, git commit + branch + dirty flag,
python/torch/transformers/sklearn versions, hostname, timestamp, full
config snapshot (max_length, batch_size, epochs, learning_rate,
torch_threads), and explicit flags
`real_pretrained_weights` / `real_human_labels` /
`weak_pretraining_dataset`. The manifests are the recommended primary
record when reviewers ask "what data and code produced this number?"

## Reproduce locally

```bash
# 1) Cache pretrained RoBERTa-base from the legacy S3 mirror.
mkdir -p models/.cache/roberta-base && cd models/.cache/roberta-base
for f in roberta-base-config.json:config.json \
         roberta-base-pytorch_model.bin:pytorch_model.bin \
         roberta-base-vocab.json:vocab.json \
         roberta-base-merges.txt:merges.txt; do
  curl -fsSL "https://s3.amazonaws.com/models.huggingface.co/bert/${f%:*}" \
    -o "${f#*:}"
done; cd -

# 2) Adapter that emits MITweet gold in project schema.
python3 scripts/mitweet_to_project_gold.py

# 3) Train B1 + B2 across 3 seeds.
for s in 42 43 44; do
  python3 scripts/b1_mitweet_real.py --seed $s \
    --output-json outputs/real_runs/b1_seed${s}/summary.json
  python3 scripts/train_mitweet_real.py --seed $s \
    --encoder-path models/.cache/roberta-base \
    --output-json outputs/real_runs/mitweet_seed${s}/summary.json \
    --output-predictions-csv outputs/real_runs/mitweet_seed${s}/val_predictions.csv
done

# 4) Aggregate + paired bootstrap + error analysis.
python3 scripts/run_seeds.py run \
  --command "echo cached seed={seed}" --seeds 42,43,44 \
  --summary-template "outputs/real_runs/mitweet_seed{seed}/summary.json" \
  --metrics "metrics.macro_f1_mean,metrics.relevance_macro_f1,metrics.economic_direction_macro_f1,metrics.social_direction_macro_f1" \
  --output-json outputs/real_runs/b2_mitweet_aggregate.json
for s in 42 43 44; do
  python3 scripts/run_seeds.py compare \
    --pred-a outputs/real_runs/b1_seed${s}/summary_val_predictions.csv \
    --pred-b outputs/real_runs/mitweet_seed${s}/val_predictions.csv \
    --gold-aggregates-csv outputs/real_runs/mitweet_gold_aggregates.csv \
    --label-a b1 --label-b b2_roberta --bootstrap 2000 \
    --output-json outputs/real_runs/compare_b1_vs_b2_seed${s}.json
  python3 scripts/error_analysis.py \
    --predictions-csv outputs/real_runs/mitweet_seed${s}/val_predictions.csv \
    --gold-aggregates-csv outputs/real_runs/mitweet_gold_aggregates.csv \
    --raw-posts-csv outputs/real_runs/mitweet_raw_posts.csv \
    --model-name b2_real_seed${s} --min-support 30 \
    --output-dir outputs/error_analysis/b2_real_seed${s}
done
```

Total CPU time: ~37 minutes for the three RoBERTa fine-tune seeds plus
~30 seconds for everything else.
