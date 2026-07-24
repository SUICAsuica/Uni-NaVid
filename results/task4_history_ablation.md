# Task 4 history-use ablation

Date: 2026-07-24

## Question

Does the trained goal-conditioned Cross-Attention model use visual history, or
does it predict the four actions from only the current frame and goal?

The final 7500-step adapter was evaluated without updating any weights. The
current frame, goal, prompt, action target, model, and teacher-forced evaluator
were held fixed while only the visual history was changed.

## Interventions

| Condition | Visual-history intervention |
| --- | --- |
| Full | Use the correct ordered history |
| Current only | Remove all visual-history frames |
| Shuffled | Randomly permute history frames with seed 42 |
| Wrong | Replace history with a different episode |
| Recent only | Keep only the most recent 64 history frames |
| Old only | Remove the most recent 64 history frames |

`Current only` removes all visual history, but the learned compressor still
emits 64 goal-conditioned tokens by attending to its learned null-history
token. It is therefore more precisely a `no visual history` intervention, not
removal of the compressor itself.

For `Wrong`, the current frame always comes from the target sample. The source
history is selected from a different episode, preferring the same goal,
history-length bin, and nearest `time_index`. The source episode's current frame
is discarded.

The originally proposed `Old history masked: remove frames older than 64` is
identical to `Recent only`. `Old only` was added as the complementary
intervention so the older and recent portions can be measured separately.

## Training-set result

The 200-sample capacity-check manifest contains 50 samples in each history bin.

| Condition | Loss | Mean action accuracy | First action | Four-action exact |
| --- | ---: | ---: | ---: | ---: |
| Full | **0.0465** | **98.625%** | **99.0%** | **95.0%** |
| Current only | 0.3283 | 88.0% | 85.0% | 68.5% |
| Shuffled | 0.0471 | 98.375% | 98.0% | 94.0% |
| Wrong | 0.4187 | 83.25% | 82.5% | 52.0% |
| Recent only | 0.2693 | 90.875% | 89.0% | 77.5% |
| Old only | 0.1529 | 93.5% | 93.0% | 82.0% |

Relative to Full:

- removing visual history reduces exact match by 26.5 percentage points;
- replacing it with another episode reduces exact match by 43 points;
- shuffling frame order reduces exact match by only 1 point;
- retaining only the recent 64 frames reduces exact match by 17.5 points; and
- retaining only frames older than the recent window reduces exact match by
  13 points.

Exact match by history length:

| Condition | Short | Medium | Long | Extra-long |
| --- | ---: | ---: | ---: | ---: |
| Full | 96% | 92% | 96% | 96% |
| Current only | 74% | 70% | 60% | 70% |
| Shuffled | 96% | 92% | 94% | 94% |
| Wrong | 56% | 58% | 54% | 40% |
| Recent only | 96% | 86% | 60% | 68% |
| Old only | 74% | 66% | 96% | 92% |

On the training samples, the model clearly uses visual-history content. Correct
history improves predictions, and incorrect history actively misleads the
model. Frames older than 64 steps account for much of the result in the long
and extra-long bins.

The shuffled result is nearly unchanged. The model does not show strong use of
frame order despite receiving temporal embeddings. A likely interpretation is
that it treats history mostly as a set of visual observations or as an episode
fingerprint rather than learning detailed temporal transitions.

## Episode-disjoint held-out result

This set contains 140 short-history samples from four unseen episodes. All
samples are chair episodes from one scene already represented in training.

| Condition | Loss | Mean action accuracy | First action | Four-action exact |
| --- | ---: | ---: | ---: | ---: |
| Full | 1.3566 | **64.82%** | **70.71%** | **12.86%** |
| Current only | **1.2716** | 61.79% | 61.43% | 12.14% |
| Shuffled | 1.3538 | 64.11% | 69.29% | 11.43% |
| Wrong | 1.3716 | **64.82%** | 67.86% | 12.14% |
| Recent only | 1.3566 | **64.82%** | **70.71%** | **12.86%** |
| Old only | **1.2716** | 61.79% | 61.43% | 12.14% |

All held-out histories are shorter than 64 frames, so `Recent only` is
identical to Full and `Old only` is identical to Current only.

Full history improves mean action accuracy by 3.04 points and first-action
accuracy by 9.29 points over Current only. It improves exact match by only 0.71
points: 18 exact samples instead of 17. It also has worse teacher-forced loss.
Wrong history has the same mean action accuracy as Full and only one fewer exact
sample.

These data do not demonstrate robust, generalizable use of correct history.
The strong history dependence observed during memorization largely disappears
on unseen episodes.

## Conclusion

The current-frame-only explanation is rejected for the 200 training samples:
the trained model uses visual-history content there. It is not rejected in a
meaningful generalization sense. On the limited held-out set, correct history
provides a first-action benefit but no reliable four-action or loss
improvement.

The combined evidence supports this narrower statement:

> The compressor can encode and use history as a memorization signal, but this
> experiment does not yet show that it learned transferable temporal
> navigation reasoning.

The near-invariance to shuffled history is also evidence that the learned
temporal embeddings are not being used strongly. The next dataset must contain
scene-disjoint medium, long, and extra-long episodes before retraining and
repeating this ablation.

## Reproduction

```bash
./scripts/task4_history_ablation.sh

.venv/bin/python tools/summarize_task4_ablations.py \
  --input-dir outputs/task4-history-ablation \
  --output results/task4_history_ablation_summary.json
```

The aggregate evaluations and all deltas from Full are recorded in
`results/task4_history_ablation_summary.json`.
