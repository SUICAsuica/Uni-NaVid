#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${DATA_ROOT:-/home/novel/uninavid-data/task4_uninavid_full}"
manifest="${MANIFEST:-${data_root}/task4_smoke_4.json}"
model_path="${MODEL_PATH:-${repo_root}/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2}"
output_dir="${OUTPUT_DIR:-${repo_root}/outputs/task4-smoke-baseline}"

cd "${repo_root}"

"${repo_root}/.venv/bin/python" tools/build_task4_subset.py \
  --samples-json "${data_root}/samples_train.json" \
  --stock-json "${data_root}/uninavid_train.json" \
  --output "${manifest}" \
  --per-bin 1 \
  --seed 42

"${repo_root}/.venv/bin/python" tools/smoke_task4_loader.py \
  --manifest "${manifest}" \
  --video-folder "${data_root}" \
  --model-path "${model_path}" \
  --image-processor "${repo_root}/uninavid/processor/clip-patch14-224"

"${repo_root}/.venv/bin/python" -m uninavid.train.train \
  --model_name_or_path "${model_path}" \
  --version imgsp_v1 \
  --data_path "${manifest}" \
  --image_folder "${data_root}" \
  --video_folder "${data_root}" \
  --vision_tower "${repo_root}/model_zoo/eva_vit_g.pth" \
  --image_processor "${repo_root}/uninavid/processor/clip-patch14-224" \
  --tune_vision_encoder False \
  --tune_mm_mlp_adapter True \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --video_fps 1 \
  --compress_type grid:2 \
  --bf16 True \
  --tf32 True \
  --output_dir "${output_dir}" \
  --max_steps 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --weight_decay 0 \
  --warmup_ratio 0 \
  --logging_steps 1 \
  --save_strategy no \
  --evaluation_strategy no \
  --model_max_length 2048 \
  --gradient_checkpointing False \
  --dataloader_num_workers 0 \
  --lazy_preprocess True \
  --report_to none
