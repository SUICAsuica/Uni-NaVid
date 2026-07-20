# Task 4 goal-conditioned history compression: 200-sample overfit check

## Setup

- Date: 2026-07-19
- Branch: `learned-history-compressor`
- Training set: 200 samples, stratified into 50 samples for each history bin
- History input: 8 x 8 = 64 tokens per frame, up to 320 observed frames
- Learned output: 64 fixed history tokens; the current frame remains separate
- Goal input: frozen Llama token embeddings, masked mean pooling, trainable goal projector
- Trainable parameters: history compressor and goal projector only
- Training: 400 steps (2 epochs), batch size 1, cosine learning rate from `1e-4`
- Evaluation: teacher-forced on the same 200 training samples, with video augmentation disabled

This is a pipeline and memorization check. It is not a held-out navigation or
online rollout evaluation.

## Training result

- Aggregate train loss: 0.7263
- First logged 10-step loss: 0.9671
- Step-200 10-step loss: 0.6953
- Final 10-step loss: 0.9165
- Best logged 10-step loss: 0.5530 at step 320

Loss decreased in several intervals, but the run did not completely memorize the
200 samples.

## Teacher-forced comparison

| Metric | Heuristic | Goal Cross-Attention | Difference |
| --- | ---: | ---: | ---: |
| Loss | 0.7073 | 0.6672 | -0.0401 |
| Mean action accuracy | 62.625% | 62.625% | 0.000 pp |
| First action accuracy | 59.0% | 57.5% | -1.5 pp |
| Four-action exact match | 17.0% | 13.0% | -4.0 pp |

Action accuracy by target position:

| Position | Heuristic | Goal Cross-Attention | Difference |
| --- | ---: | ---: | ---: |
| 1 | 59.0% | 57.5% | -1.5 pp |
| 2 | 63.0% | 64.0% | +1.0 pp |
| 3 | 67.0% | 70.0% | +3.0 pp |
| 4 | 61.5% | 59.0% | -2.5 pp |

## Conclusion

The learned compressor reduced teacher-forced loss by 0.0401, but it did not
beat the heuristic baseline on mean action accuracy or exact match after 400
steps. Step 8 is therefore complete as an experiment, with a negative accuracy
result. The next experiment should change the optimization schedule or training
budget before claiming an improvement, then validate on held-out episodes and
online rollouts.

Machine-readable metrics are in `results/task4_overfit_200_comparison.json`.

## Follow-up: one layer and 2000 steps

The follow-up run changed only the optimization and compressor-depth settings:

- Cross-Attention decoder layers: 1
- Dropout: 0
- Video augmentation: disabled
- Learning rate: constant `3e-4`, no warmup
- Training: 2000 steps (10 epochs)

The mean logged loss by epoch decreased monotonically from 1.0115 to 0.5771.
Aggregate train loss was 0.6564. This confirms that the one-layer compressor can
learn, but it still did not memorize the 200 samples.

| Metric | Heuristic | 2 layers, 400 steps | 1 layer, 2000 steps |
| --- | ---: | ---: | ---: |
| Loss | 0.7073 | 0.6672 | 0.5637 |
| Mean action accuracy | 62.625% | 62.625% | 68.0% |
| First action accuracy | 59.0% | 57.5% | 60.0% |
| Four-action exact match | 17.0% | 13.0% | 18.0% |

Compared with the heuristic, the one-layer run improved mean action accuracy by
5.375 percentage points and exact match by 1 point. The largest mean action
accuracy gains were in the medium (+8.0 points) and long (+7.5 points) history
bins. Exact match improved only in the medium bin; it fell in the other three
bins.

This supports using one layer for the next diagnostic experiment. It does not
show that one layer is sufficient for the final model: 18% exact match is far
below the 95% memorization target, and this remains a teacher-forced evaluation
on the training samples.

The follow-up command was:

```bash
OUTPUT_DIR=outputs/task4-overfit-200-goal-1layer-2000 \
MAX_STEPS=2000 \
HISTORY_NUM_LAYERS=1 \
HISTORY_DROPOUT=0 \
VIDEO_AUGMENTATION=False \
LEARNING_RATE=3e-4 \
WARMUP_RATIO=0 \
LR_SCHEDULER_TYPE=constant \
./scripts/task4_overfit_200_goal.sh
```

Machine-readable follow-up metrics are in
`results/task4_overfit_200_1layer_2000_comparison.json`.
