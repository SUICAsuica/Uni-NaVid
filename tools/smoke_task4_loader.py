#!/usr/bin/env python3
"""Exercise Uni-NaVid's stock loader against a Task 4 manifest."""

import argparse
from pathlib import Path

from transformers import AutoTokenizer, CLIPImageProcessor

from uninavid.train.train import (
    DataArguments,
    DataCollatorForSupervisedDataset,
    LazySupervisedDataset,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-folder", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image-processor", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    tokenizer.model_max_length = 2048
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.unk_token

    data_args = DataArguments(
        data_path=str(args.manifest),
        lazy_preprocess=True,
        is_multimodal=True,
        video_folder=str(args.video_folder),
        video_fps=1,
        image_aspect_ratio="pad",
    )
    data_args.image_processor = CLIPImageProcessor.from_pretrained(args.image_processor)
    data_args.mm_use_im_start_end = False

    dataset = LazySupervisedDataset(
        data_path=str(args.manifest), tokenizer=tokenizer, data_args=data_args
    )
    instances = [dataset[index] for index in range(len(dataset))]
    collator = DataCollatorForSupervisedDataset(tokenizer)
    batch = collator(instances[:1])

    frame_counts = [instance["image"].shape[0] for instance in instances]
    supervised_tokens = [int((instance["labels"] != -100).sum()) for instance in instances]
    print(f"samples={len(instances)}")
    print(f"frame_counts={frame_counts}")
    print(f"supervised_tokens={supervised_tokens}")
    print(f"batch_input_shape={tuple(batch['input_ids'].shape)}")
    print(f"batch_video_shape={tuple(batch['images'][0].shape)}")


if __name__ == "__main__":
    main()
