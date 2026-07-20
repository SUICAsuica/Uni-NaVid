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

## Follow-up: jointly train the multimodal projector

The next controlled run kept the one-layer, 2000-step configuration and also
trained the existing multimodal projector:

- Cross-Attention decoder layers: 1
- Dropout: 0
- Video augmentation: disabled
- Learning rate: constant `3e-4`, no warmup
- Training: 2000 steps (10 epochs)
- Trainable parameters: history compressor, goal projector, and `mm_projector`

The saved adapter contains 36 history-compressor tensors and 4 multimodal-
projector tensors, with no LLM or vision-encoder weights. Mean logged loss by
epoch decreased from 0.8368 in epoch 1 to 0.7130 in epoch 10, but evaluation was
worse than both comparison methods.

| Metric | Heuristic | Compressor only | Compressor + projector |
| --- | ---: | ---: | ---: |
| Loss | 0.7073 | 0.5637 | 0.7072 |
| Mean action accuracy | 62.625% | 68.0% | 56.5% |
| First action accuracy | 59.0% | 60.0% | 51.5% |
| Four-action exact match | 17.0% | 18.0% | 8.5% |

Jointly updating the projector at the same `3e-4` learning rate did not reproduce
the 20-sample memorization result at 200 samples. The projector is shared by the
compressed history and the uncompressed current frame, so changing it can also
damage the pretrained current-frame interface. A useful next diagnostic is to
use separate learning rates: keep the compressor at `3e-4` and update the
projector conservatively, for example at `1e-5`, preferably from the successful
compressor-only checkpoint. Adding Q-Former depth or LLM LoRA is not justified
until this interface effect is isolated.

Machine-readable metrics are in
`results/task4_overfit_200_with_projector_comparison.json`.

## Follow-up: lower learning rate for the multimodal projector

The final controlled run used separate optimizer groups while keeping all other
settings unchanged:

- History compressor learning rate: constant `3e-4`
- Multimodal projector learning rate: constant `1e-5`
- Cross-Attention decoder layers: 1
- Dropout: 0
- Video augmentation: disabled
- Training: 2000 steps (10 epochs), seed 42

Mean logged loss decreased monotonically by epoch from 0.7173 to 0.5109. The
aggregate train loss was 0.6030, and the minimum logged 10-step loss was 0.2987.

| Metric | Heuristic | Compressor only | Projector at `3e-4` | Projector at `1e-5` |
| --- | ---: | ---: | ---: | ---: |
| Loss | 0.7073 | 0.5637 | 0.7072 | **0.4629** |
| Mean action accuracy | 62.625% | 68.0% | 56.5% | **77.5%** |
| First action accuracy | 59.0% | 60.0% | 51.5% | **76.0%** |
| Four-action exact match | 17.0% | 18.0% | 8.5% | **41.0%** |

The lower projector learning rate improved mean action accuracy by 9.5 points
and exact match by 23 points over the compressor-only run. This supports the
interface-damage hypothesis: the projector benefits from adaptation, but its
pretrained mapping is damaged when it is updated at the compressor's learning
rate. The run still falls short of the 95% memorization target and remains a
teacher-forced evaluation on training samples. The next step is a longer
low-projector-LR overfit run or checkpointed evaluation before moving to held-out
episodes.

The run used:

```bash
OUTPUT_DIR=outputs/task4-overfit-200-projector-lr1e-5 \
MAX_STEPS=2000 \
HISTORY_NUM_LAYERS=1 \
HISTORY_DROPOUT=0 \
VIDEO_AUGMENTATION=False \
LEARNING_RATE=3e-4 \
WARMUP_RATIO=0 \
LR_SCHEDULER_TYPE=constant \
TUNE_MM_MLP_ADAPTER=True \
LR_MULTI='mm_projector:0.03333333333333333' \
./scripts/task4_overfit_200_goal.sh
```

Machine-readable metrics are in
`results/task4_overfit_200_projector_lr1e-5_comparison.json`.
