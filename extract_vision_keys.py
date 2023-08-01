#!/usr/bin/env python
# coding=utf-8
"""
Extracting the vision features as the keys in retrieval.
"""

import argparse
import os

import h5py
import numpy
import torch
from accelerate import Accelerator
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from torchvision.io import ImageReadMode, read_image
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, set_seed

from contrastive_train import Transform
from model.modeling_zrigf import ZRIGFForConditionalGeneration

torch.backends.cuda.matmul.allow_tf32 = True


IMAGE_SETS = ["open_images", "coco", "image_chat"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default=None, required=True,
                        help='Path to pretrained model')
    parser.add_argument('--image_names', type=str, default=None, required=True,
                        help='The splits of images to be extracted')
    parser.add_argument('--image_column', type=str, default=None, required=True,
                        help='The name of the column in the datasets containing the full image file paths.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='The directory to save the extracted feature keys')
    parser.add_argument('--per_device_batch_size', type=int, default=512)
    parser.add_argument('--preprocessing_num_workers', type=int, default=None)
    args = parser.parse_args()

    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        print(accelerator.state)

    set_seed(42)

    # Load the model
    print('Here is loading the trained vision model part-----')
    image_processor = AutoImageProcessor.from_pretrained(args.model_name_or_path)
    model = ZRIGFForConditionalGeneration.from_pretrained(args.model_name_or_path)
    vision_config = model.model.vision_encoder.config

    # force to use the default image transformation
    image_transformations = Transform(
        vision_config.image_size, image_processor.image_mean, image_processor.image_std
    )
    # image_transformations = torch.jit.script(image_transformations)

    # Preprocessing the datasets.
    def transform_images(examples):
        images = [read_image(image_file, mode=ImageReadMode.RGB) for image_file in examples[args.image_column]]
        examples["pixel_values"] = [image_transformations(image) for image in images]
        return examples

    def filter_corrupt_images(examples):
        """remove problematic images"""
        valid_images = []
        for image_file in examples[args.image_column]:
            try:
                # Image.open(image_file)
                read_image(image_file, mode=ImageReadMode.RGB)
                valid_images.append(True)
            except Exception:
                valid_images.append(False)
        return valid_images

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        image_path = [example["image_path"] for example in examples]
        return {
            "pixel_values": pixel_values,
            "image_path": image_path,
        }

    for image_name in args.image_names.split(','):
        assert image_name in IMAGE_SETS, f"{image_name} not in {IMAGE_SETS}"

        # Load dataset
        raw_dataset = load_dataset(f"data/{image_name}")
        dataset = concatenate_datasets([raw_dataset[split] for split in raw_dataset])
        dataset = dataset.remove_columns([col for col in dataset.column_names if col != args.image_column])

        with accelerator.main_process_first():
            print("Preprocessing the datasets...")
            dataset = dataset.filter(
                filter_corrupt_images, batched=True, num_proc=args.preprocessing_num_workers * 2,
            )
            # Transform images on the fly as doing it on the whole dataset takes too much time.
            dataset.set_transform(transform_images)

        dataloader = DataLoader(
            dataset, batch_size=args.per_device_batch_size, num_workers=args.preprocessing_num_workers,
            collate_fn=collate_fn, drop_last=True, pin_memory=True,
        )

        model, dataloader = accelerator.prepare(model, dataloader)

        image_features = []
        image_paths = []
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
        model.eval()
        for step, batch in enumerate(dataloader):
            with torch.no_grad():
                image_feature = accelerator.unwrap_model(model).get_image_features(
                    pixel_values=batch["pixel_values"],
                )
                image_path = batch["image_path"]
                pixel_value = batch["pixel_values"].cpu().numpy()
                image_features.extend(accelerator.gather(image_feature).cpu().numpy())

                if accelerator.num_processes > 1:
                    output = [None for _ in range(accelerator.num_processes)]
                    torch.distributed.all_gather_object(output, image_path)
                    image_path = numpy.concatenate(output).tolist()
                image_paths.extend(image_path)

                progress_bar.update(1)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            output_dir = args.output_dir
            if args.output_dir is None:
                output_dir = args.model_name_or_path + '/keys'  # Save the keys with the model dict
            os.makedirs(output_dir, exist_ok=True)
            h5_path = os.path.join(output_dir, f"{image_name}.hdf5")
            with h5py.File(h5_path, "w") as f:
                f["image_paths"] = image_paths
                f["image_features"] = image_features
