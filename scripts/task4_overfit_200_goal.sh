#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${DATA_ROOT:-/home/novel/uninavid-data/task4_uninavid_full}"
manifest="${MANIFEST:-${data_root}/task4_overfit_200_goal.json}"
model_path="${MODEL_PATH:-${repo_root}/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2}"
output_dir="${OUTPUT_DIR:-${repo_root}/outputs/task4-overfit-200-goal}"
max_steps="${MAX_STEPS:-400}"

cd "${repo_root}"

"${repo_root}/.venv/bin/python" -m uninavid.train.train \
  --model_name_or_path "${model_path}" \
  --version imgsp_v1 \
  --data_path "${manifest}" \
  --image_folder "${data_root}" \
  --video_folder "${data_root}" \
  --vision_tower "${repo_root}/model_zoo/eva_vit_g.pth" \
  --image_processor "${repo_root}/uninavid/processor/clip-patch14-224" \
  --video_augmentation True \
  --tune_vision_encoder False \
  --tune_mm_mlp_adapter False \
  --history_compressor_type cross_attention \
  --history_num_queries 64 \
  --history_hidden_size 512 \
  --history_num_heads 8 \
  --history_num_layers 2 \
  --history_ffn_dim 2048 \
  --history_dropout 0.1 \
  --history_max_frames 512 \
  --history_goal_conditioned True \
  --tune_history_compressor True \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --compress_type grid:2 \
  --video_fps 1 \
  --bf16 True \
  --tf32 True \
  --output_dir "${output_dir}" \
  --max_steps "${max_steps}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --save_strategy no \
  --evaluation_strategy no \
  --model_max_length 2048 \
  --gradient_checkpointing False \
  --dataloader_num_workers 2 \
  --lazy_preprocess True \
  --report_to none \
  --seed 42
