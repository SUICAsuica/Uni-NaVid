# Task 4: final 200-sample capacity check and held-out evaluation

Date: 2026-07-21

## Conclusion

The goal-conditioned one-layer Cross-Attention compressor reached the target
of 95% four-action exact match on the 200 training samples. This establishes
that the implemented model has enough capacity to memorize this pilot task.

It did not outperform the Uni-NaVid heuristic on the episode-disjoint held-out
set. Cross-Attention improved first-action accuracy, but had worse loss, mean
action accuracy, and four-action exact match. The result therefore does not
support a generalization improvement.

## Training-set capacity check

The final adapter is the initial 2000-step run followed by 5500 continuation
steps, for 7500 optimizer steps in total. The compressor used one layer and no
dropout. The history compressor learning rate was `3e-4`; the multimodal
projector was jointly tuned at `1e-5`. EVA-CLIP and the language model remained
frozen.

| Metric | Result |
| --- | ---: |
| Teacher-forced loss | 0.0465 |
| Action accuracy, positions 1-4 | 99.0%, 98.5%, 98.5%, 98.5% |
| Mean action accuracy | 98.625% |
| First-action accuracy | 99.0% |
| Four-action exact match | **95.0%** |

Exact match by history length:

| History bin | Samples | Exact match | Failed samples |
| --- | ---: | ---: | ---: |
| Short, `t < 64` | 50 | 96.0% | 2 |
| Medium, `64 <= t < 128` | 50 | 92.0% | 4 |
| Long, `128 <= t < 256` | 50 | 96.0% | 2 |
| Extra-long, `t >= 256` | 50 | 96.0% | 2 |

Failures do not increase with history length. The medium bin contains 4 of the
10 failures and is the weakest bin, so there is a mild medium-history
concentration that should be checked again on a larger dataset.

The machine-readable evaluation is in `task4_overfit_200_step5500.json`.

## Held-out construction

The original pilot contains 2,612 samples from 23 successful episodes. The
balanced 200-sample training manifest touches 19 of those episodes. Every
sample from those 19 episodes was excluded, leaving all 140 samples from the
remaining four episodes. Training/held-out ID overlap and episode overlap are
both zero.

```bash
.venv/bin/python tools/build_task4_subset.py \
  --samples-json /home/novel/uninavid-data/task4_uninavid_full/samples_train.json \
  --stock-json /home/novel/uninavid-data/task4_uninavid_full/uninavid_train_goal.json \
  --exclude-manifest /home/novel/uninavid-data/task4_uninavid_full/task4_overfit_200_goal.json \
  --all-eligible \
  --seed 42 \
  --output /home/novel/uninavid-data/task4_uninavid_full/task4_heldout_episode_disjoint_goal.json
```

Held-out manifest SHA-256:
`ac9138c9e434ee2a87fcf1b36e7dace3ecb2565b789824b4e4aca9f2a5e00031`.

The held-out data is episode-disjoint but not scene-disjoint. All four episodes
come from one scene already represented in training, all use the `chair` goal,
and all 140 samples are short histories with `time_index` from 0 through 50.
There were no untouched medium, long, or extra-long episodes in this pilot.
This evaluation therefore measures only short-history transfer to unseen
episodes, not broad ObjectNav or long-history generalization.

## Held-out comparison

Both modes were evaluated on the same manifest with the same base model, vision
encoder, 64-token history budget, prompts, targets, and teacher-forced metric
implementation. The exact evaluator inputs were:

```bash
common_args=(
  --manifest /home/novel/uninavid-data/task4_uninavid_full/task4_heldout_episode_disjoint_goal.json
  --video-folder /home/novel/uninavid-data/task4_uninavid_full
  --model-path model_zoo/uninavid-7b-full-224-video-fps-1-grid-2
  --vision-tower model_zoo/eva_vit_g.pth
  --image-processor uninavid/processor/clip-patch14-224
  --history-num-layers 1
  --history-dropout 0
)

.venv/bin/python tools/evaluate_task4.py "${common_args[@]}" \
  --mode heuristic \
  --output outputs/task4-overfit-200-projector-lr1e-5-continued-2000/eval_heldout_episode_disjoint_heuristic.json

.venv/bin/python tools/evaluate_task4.py "${common_args[@]}" \
  --mode goal_cross_attention \
  --adapter outputs/task4-overfit-200-projector-lr1e-5-continued-2000/multimodal_adapters.bin \
  --output outputs/task4-overfit-200-projector-lr1e-5-continued-2000/eval_heldout_episode_disjoint_cross_attention.json
```

| Metric | Heuristic | Cross-Attention | Cross minus heuristic |
| --- | ---: | ---: | ---: |
| Teacher-forced loss | **0.7639** | 1.3566 | +0.5926 |
| Action accuracy, position 1 | 62.14% | **70.71%** | +8.57 pp |
| Action accuracy, position 2 | **70.71%** | 67.86% | -2.86 pp |
| Action accuracy, position 3 | **69.29%** | 64.29% | -5.00 pp |
| Action accuracy, position 4 | **67.14%** | 56.43% | -10.71 pp |
| Mean action accuracy | **67.32%** | 64.82% | -2.50 pp |
| Four-action exact match | **25.0%** | 12.86% | -12.14 pp |

The Cross-Attention model transfers better for the immediate action but loses
accuracy at later action positions, producing substantially fewer exact
four-action sequences. The higher loss also indicates worse calibration on
these unseen episodes.

This comparison is operationally matched but does not isolate only the
compressor: the Cross-Attention adapter includes the jointly trained
multimodal projector, while the heuristic uses the stock projector. A future
controlled ablation should hold projector weights fixed or evaluate both
compressors with a separately shared projector.

Machine-readable metrics and deltas are in
`task4_heldout_episode_disjoint_comparison.json`.

Final adapter SHA-256:
`712c014bcb802af25eb7713c63790d0b35865426e1c0f49b55a1e756e2b5d7fe`.

## Required next experiment

Collect additional successful episodes before drawing a generalization
conclusion. The next split must reserve entire scenes or, at minimum, enough
complete episodes to cover all four history-length bins and multiple goals.
