#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${DATA_ROOT:-/home/novel/uninavid-data/task4_uninavid_full}"
model_path="${MODEL_PATH:-${repo_root}/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2}"
adapter="${ADAPTER:-${repo_root}/outputs/task4-overfit-200-projector-lr1e-5-continued-2000/multimodal_adapters.bin}"
output_dir="${OUTPUT_DIR:-${repo_root}/outputs/task4-history-ablation}"
ablations="${ABLATIONS:-full current_only shuffled wrong recent_only old_only}"

common_args=(
  --video-folder "${data_root}"
  --model-path "${model_path}"
  --vision-tower "${repo_root}/model_zoo/eva_vit_g.pth"
  --image-processor "${repo_root}/uninavid/processor/clip-patch14-224"
  --mode goal_cross_attention
  --adapter "${adapter}"
  --history-num-layers 1
  --history-dropout 0
  --ablation-seed 42
)

mkdir -p "${output_dir}"
cd "${repo_root}"

for split in train heldout; do
  if [[ "${split}" == "train" ]]; then
    manifest="${data_root}/task4_overfit_200_goal.json"
  else
    manifest="${data_root}/task4_heldout_episode_disjoint_goal.json"
  fi

  for ablation in ${ablations}; do
    echo "split=${split} history_ablation=${ablation}"
    "${repo_root}/.venv/bin/python" tools/evaluate_task4.py \
      "${common_args[@]}" \
      --manifest "${manifest}" \
      --history-ablation "${ablation}" \
      --output "${output_dir}/${split}_${ablation}.json"
  done
done
