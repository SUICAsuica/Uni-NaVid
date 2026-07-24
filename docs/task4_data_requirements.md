# Task 4 additional data requirements

Updated: 2026-07-24

## Purpose

The next dataset must determine whether learned Cross-Attention history
compression generalizes better than Uni-NaVid's heuristic compression. The
current pilot is sufficient for implementation and capacity checks, but not
for this comparison:

| Property | Current pilot |
| --- | ---: |
| Samples | 2,612 |
| Successful episodes | 23 |
| Scenes | 2 HM3D train scenes |
| Goals | chair and toilet |
| Episode-disjoint held-out set | 4 episodes, 140 samples |
| Held-out history coverage | short only, `time_index` 0-50 |

All current held-out episodes are chair episodes from one scene that also
appears in training. The next collection must fix scene, goal, episode, and
history-length coverage rather than only increase the sample count.

## Reuse of the current data

Do not discard or regenerate the existing 2,612 samples. Assign both current
scenes to `train` and keep their 23 successful episodes as training data. They
count toward the train minimum below.

Collect additional train scenes to fill the missing goals and history bins.
Collect validation and test data from entirely different scenes. None of the
current samples may be moved into validation or test because their scenes have
already been used for training and hyperparameter development.

## Unit of collection

An episode is the indivisible collection and split unit. Store one complete
video or one complete EVA feature tensor per episode, plus the action sequence
and metadata. Generate prefix samples through `episode + time_index` references;
do not create a separate copied video for every prefix.

Each episode record must contain:

- stable `episode_id`;
- `scene_id` and Habitat dataset split;
- ObjectNav goal category;
- RGB observation before each action;
- action at each step and the next four actions where available;
- episode length;
- success, SPL, final distance-to-goal, and collision count;
- start position and rotation;
- collection code version, configuration, and random seed; and
- validation status for frame/action alignment.

Retain failed rollouts and their metrics for collection audits and online
evaluation. The initial imitation-training manifest may remain restricted to
validated successful trajectories so that failure behavior is not silently
used as a target.

## Required goal coverage

Collect all six ObjectNav goal families used by this codebase:

```text
chair
sofa
plant / potted plant
bed
toilet
tv_monitor / tv
```

Normalize aliases when building manifests, but preserve the original category
in episode metadata. No single goal should provide more than 30% of the
training or held-out samples.

## Required history coverage

Use the existing bins:

| Bin | Time index |
| --- | --- |
| Short | `0 <= t < 64` |
| Medium | `64 <= t < 128` |
| Long | `128 <= t < 256` |
| Extra-long | `t >= 256` |

For the next offline comparison, every held-out bin must contain:

- at least 200 prefix samples;
- at least 10 distinct episodes;
- at least 3 distinct scenes; and
- at least 3 goal categories.

Counts must be reported at all four levels: samples, episodes, scenes, and
goals. A long episode produces many correlated prefixes, so 200 prefixes from
one trajectory do not satisfy this requirement.

To obtain extra-long data, explicitly continue collection until at least ten
successful held-out episodes exceed 259 actions. Four future actions are
required after the selected `time_index`, so the final usable prefix begins at
least four actions before episode termination.

## Split policy

Choose splits before training or hyperparameter selection.

1. `train`: HM3D train scenes used to optimize the compressor and projector.
2. `validation`: different HM3D train scenes used for checkpoint and
   hyperparameter selection.
3. `test`: HM3D validation scenes used once for the final comparison.

Scene sets must be disjoint. Episode and sample IDs must therefore also be
disjoint. Do not randomly split prefix samples.

If the available environment cannot yet support a scene-disjoint test split,
create an episode-disjoint development split and label it `development`, not
`test`. Do not claim scene generalization from it.

## Minimum collection gate

Before starting the next full training run, require:

| Split | Minimum scenes | Minimum successful episodes | Goal coverage |
| --- | ---: | ---: | --- |
| Train | 6 | 60 | all 6 goals |
| Validation | 3 | 30 | at least 4 goals |
| Test | 3 | 30 | at least 4 goals |

These are minimum engineering gates, not a claim of statistical sufficiency.
Prefer at least five evaluation scenes and report confidence intervals
clustered or bootstrapped by episode and scene.

The history-coverage requirements above are additional. If 30 test episodes do
not produce ten extra-long successes, collect more episodes rather than filling
the bin with additional prefixes from the same trajectories.

## Sampling policy

Build training batches with approximately equal representation from the four
history bins. Cap the number of prefixes sampled from any single episode so a
few long trajectories cannot dominate optimization.

For validation and test:

- freeze the manifest before evaluation;
- evaluate both compressors on exactly the same prefixes;
- report the complete set as well as each history bin;
- report per-goal and per-scene results;
- keep the 64-token history budget identical; and
- use episode- or scene-level uncertainty estimates.

## Controlled comparison

The next experiment must separate compression from projector adaptation:

1. Heuristic compression with a fixed shared projector.
2. Goal-conditioned Cross-Attention compression with the same fixed projector.
3. Optional second comparison where both methods receive the same projector
   training budget.

Keep EVA-CLIP, the language model, prompts, action targets, current-frame path,
token budget, and evaluation code fixed. Measure teacher-forced loss,
per-position action accuracy, first-action accuracy, four-action exact match,
latency, and peak memory.

After offline selection, compare online ObjectNav Success Rate, SPL,
distance-to-goal, and collision rate on the frozen test scenes.

## Validation before training

Reject or repair a collection if any of the following checks fail:

- duplicated episode, sample, or scene assignment across splits;
- missing frames, corrupt video, or inconsistent frame rate;
- frame/action count mismatch;
- unknown or inconsistent goal labels;
- fewer than four future actions for a generated target;
- missing success/SPL metadata;
- a history bin below its sample, episode, scene, or goal minimum; or
- one episode or goal exceeding the configured sampling cap.

Record a machine-readable summary containing counts by split, history bin,
scene, goal, success status, and action class. Store the manifest checksum and
the collection/configuration commit with every experiment result.

## Existing collection commands

Run collection from the L3MVN handoff repository with its documented Habitat
2022/HM3D v0.1/ObjectNav v1 environment. Do not mix ObjectNav v2 data into this
unported stack.

```bash
python main_llm_vis.py \
  --split train --eval 1 --auto_gpu_config 0 \
  --total_num_scenes 6 -n 1 --num_eval_episodes 100 \
  --load pretrained_models/llm_model.pt --use_gtsem 1 \
  --num_local_steps 10 --collect_rollouts --uninavid_four_actions \
  --rollout_keep_failures \
  --rollout_dir objectnav_dataset/trajectories

python tools/validate_rollouts.py \
  --trajectory-root objectnav_dataset/trajectories

python tools/build_uninavid_samples.py \
  --trajectory-root objectnav_dataset/trajectories \
  --output-root objectnav_dataset
```

The numerical collection arguments are a starting batch, not proof that the
minimum gates have been met. Inspect the generated summary and repeat
collection until the scene, episode, goal, and history requirements all pass.
