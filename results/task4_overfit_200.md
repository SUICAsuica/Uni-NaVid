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
