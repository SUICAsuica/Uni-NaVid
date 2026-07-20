#!/usr/bin/env python3
"""Teacher-forced offline evaluation for Task 4 navigation manifests."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoConfig, AutoTokenizer, CLIPImageProcessor

from uninavid.model import LlavaLlamaAttForCausalLM
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
    parser.add_argument("--vision-tower", type=Path, required=True)
    parser.add_argument("--image-processor", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("heuristic", "cross_attention", "goal_cross_attention"),
        required=True,
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--history-num-layers", type=int, default=2)
    parser.add_argument("--history-dropout", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def history_bin(time_index):
    if time_index < 64:
        return "short"
    if time_index < 128:
        return "medium"
    if time_index < 256:
        return "long"
    return "extra_long"


def load_multimodal_adapter(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    compressor_prefix = "model.history_compressor."
    compressor_state = {
        key[len(compressor_prefix):]: value
        for key, value in state.items()
        if key.startswith(compressor_prefix)
    }
    if not compressor_state:
        raise ValueError(f"No history compressor weights found in {path}")
    model.get_model().history_compressor.load_state_dict(compressor_state)

    projector_prefix = "model.mm_projector."
    projector_state = {
        key[len(projector_prefix):]: value
        for key, value in state.items()
        if key.startswith(projector_prefix)
    }
    if projector_state:
        model.get_model().mm_projector.load_state_dict(projector_state)


def build_model(args, device):
    learned = args.mode != "heuristic"
    goal_conditioned = args.mode == "goal_cross_attention"
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = LlavaLlamaAttForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    model_args = SimpleNamespace(
        vision_tower=str(args.vision_tower),
        image_processor=str(args.image_processor),
        mm_vision_select_layer=-2,
        mm_vision_select_feature="patch",
        pretrain_mm_mlp_adapter=None,
        mm_projector_type="mlp2x_gelu",
        compress_type="grid:2",
        run_type="train",
        history_compressor_type="cross_attention" if learned else "heuristic",
        history_num_queries=64,
        history_hidden_size=512,
        history_num_heads=8,
        history_num_layers=args.history_num_layers,
        history_ffn_dim=2048,
        history_dropout=args.history_dropout,
        history_max_frames=512,
        history_goal_conditioned=goal_conditioned,
    )
    model.get_model().initialize_vision_modules(model_args=model_args, max_token=2048)
    if args.adapter is not None:
        if not learned:
            raise ValueError("--adapter is only valid for learned compressor modes")
        load_multimodal_adapter(model, args.adapter)
    elif learned:
        raise ValueError("Learned compressor modes require --adapter")

    model.config.mm_use_im_start_end = False
    model.config.mm_use_im_patch_token = False
    model.config.image_aspect_ratio = "pad"
    model.to(device=device, dtype=torch.bfloat16)
    if learned:
        model.get_model().history_compressor.to(
            device=device,
            dtype=torch.bfloat16,
        )
    model.eval()
    return model


def make_dataset(args, tokenizer, image_processor):
    data_args = DataArguments(
        data_path=str(args.manifest),
        lazy_preprocess=True,
        is_multimodal=True,
        video_folder=str(args.video_folder),
        video_fps=1,
        image_aspect_ratio="pad",
        video_augmentation=False,
    )
    data_args.image_processor = image_processor
    data_args.mm_use_im_start_end = False
    dataset = LazySupervisedDataset(
        data_path=str(args.manifest),
        tokenizer=tokenizer,
        data_args=data_args,
    )
    return dataset, DataCollatorForSupervisedDataset(tokenizer)


def new_metrics():
    return {
        "samples": 0,
        "loss_sum": 0.0,
        "position_correct": [0, 0, 0, 0],
        "exact_match": 0,
    }


def update_metrics(metrics, loss, predictions, labels):
    metrics["samples"] += 1
    metrics["loss_sum"] += loss
    for index in range(4):
        metrics["position_correct"][index] += int(
            predictions[index] == labels[index]
        )
    metrics["exact_match"] += int(predictions[:4] == labels[:4])


def finalize_metrics(metrics):
    samples = metrics["samples"]
    return {
        "samples": samples,
        "loss": metrics["loss_sum"] / samples,
        "action_accuracy": [value / samples for value in metrics["position_correct"]],
        "first_action_accuracy": metrics["position_correct"][0] / samples,
        "four_action_exact_match": metrics["exact_match"] / samples,
    }


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(device=device, dtype=torch.bfloat16)
        return value.to(device=device)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def main():
    args = parse_args()
    device = torch.device("cuda")
    manifest = json.loads(args.manifest.read_text())
    if args.max_samples is not None:
        manifest = manifest[:args.max_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    tokenizer.model_max_length = 2048
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.unk_token
    image_processor = CLIPImageProcessor.from_pretrained(args.image_processor)
    model = build_model(args, device)
    dataset, collator = make_dataset(args, tokenizer, image_processor)

    totals = new_metrics()
    by_bin = defaultdict(new_metrics)
    for index, metadata in enumerate(manifest):
        batch = collator([dataset[index]])
        batch = {key: move_to_device(value, device) for key, value in batch.items()}
        with torch.inference_mode():
            output = model(**batch)

        shifted_labels = batch["labels"][:, 1:]
        labels = shifted_labels[shifted_labels.ne(-100)].tolist()
        shifted_predictions = output.logits[:, :-1].argmax(dim=-1)
        predictions = shifted_predictions[0, -len(labels):].tolist()
        if len(labels) < 4:
            raise ValueError(f"Sample {metadata['id']} has fewer than four targets")

        loss = float(output.loss)
        update_metrics(totals, loss, predictions, labels)
        update_metrics(
            by_bin[history_bin(metadata["time_index"])],
            loss,
            predictions,
            labels,
        )
        if (index + 1) % 10 == 0:
            print(f"evaluated={index + 1}/{len(manifest)}")

    result = {
        "mode": args.mode,
        "manifest": str(args.manifest),
        "overall": finalize_metrics(totals),
        "by_history_bin": {
            name: finalize_metrics(metrics)
            for name, metrics in sorted(by_bin.items())
        },
        "teacher_forced": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
