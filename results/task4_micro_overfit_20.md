# Task 4: 20-sample micro-overfit diagnostic

## Question

Can a one-layer goal-conditioned history compressor memorize a small dataset on
its own, or does the fixed multimodal projector constrain it?

## Setup

- 20 training samples: 5 from each history-length bin
- Goals: chair and toilet
- History range: 20 to 272 frames
- Cross-Attention decoder: 1 layer, 64 output queries, dropout 0
- Video augmentation: disabled
- Optimization: 1000 steps (50 epochs), constant learning rate `3e-4`
- LLM and vision encoder: frozen in both runs
- Evaluation: teacher-forced on the same 20 training samples

Run A trained only the history compressor and goal projector. Run B also trained
the existing `mm_projector`, which maps compressed and current-frame vision
features into the frozen LLM embedding space.

## Results

| Metric | A: compressor only | B: compressor + mm_projector |
| --- | ---: | ---: |
| Evaluation loss | 0.3751 | 0.0060 |
| Mean action accuracy | 83.75% | 100.0% |
| First action accuracy | 80.0% | 100.0% |
| Four-action exact match | 50.0% | 100.0% |
| Last logged train loss | 0.4621 | 0.0378 |
| Minimum logged train loss | 0.3134 | 0.0001 |

Run B reached 100% four-action exact match in every history-length bin. Its saved
adapter contains 36 history-compressor tensors and 4 multimodal-projector tensors,
with no LLM or vision-encoder weights.

## Conclusion

One Cross-Attention layer has enough capacity to memorize this diagnostic set.
The fixed `mm_projector` was the limiting interface: training it alongside the
compressor changed exact match from 50% to 100%. This does not establish
generalization because the evaluation uses the training samples with teacher
forcing.

The next controlled experiment should train the compressor and `mm_projector` on
the 200-sample subset, then compare it with the compressor-only 2000-step result
and the heuristic baseline. LoRA on the LLM is not yet justified by this result.

## Commands

The subset was generated with:

```bash
.venv/bin/python tools/build_task4_subset.py \
  --samples-json /home/novel/uninavid-data/task4_uninavid_full/samples_train.json \
  --stock-json /home/novel/uninavid-data/task4_uninavid_full/uninavid_train_goal.json \
  --output /home/novel/uninavid-data/task4_uninavid_full/task4_micro_overfit_20_goal.json \
  --per-bin 5 \
  --seed 42
```

Both runs used `scripts/task4_overfit_200_goal.sh`. Run B added:

```bash
TUNE_MM_MLP_ADAPTER=True
```

Machine-readable metrics are in
`results/task4_micro_overfit_20_projector_comparison.json`.
