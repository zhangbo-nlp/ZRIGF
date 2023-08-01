#!/usr/bin/env python
# coding=utf-8
"""
Vokenizing dialogue corpus.
"""

import argparse
import json
import os

import h5py
import numpy
import torch
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding, set_seed

from model.modeling_zrigf import ZRIGFForConditionalGeneration

torch.backends.cuda.matmul.allow_tf32 = True

IMAGE_SETS = ["open_images", "coco"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, required=True,
                        help='Path to pretrained model')
    parser.add_argument('--corpus_names', type=str, default=None, required=True)
    parser.add_argument('--image_names', type=str, default='open_images', required=True)
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
    print('Here is loading the model part-----')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = ZRIGFForConditionalGeneration.from_pretrained(args.model_path)
    model = accelerator.prepare(model)

    image_names = args.image_names.split(",")
    keys_dir = os.path.join(args.model_path, "keys")
    image_features = []
    image_paths = []
    for image_name in image_names:
        assert image_name in IMAGE_SETS, f"{image_name} not in {IMAGE_SETS}"
        with h5py.File(os.path.join(keys_dir, f"{image_name}.hdf5"), "r") as h5_file:
            image_features.extend(h5_file["image_features"][:])
            image_paths.extend(h5_file["image_paths"].asstr()[:])

    image_dataset = Dataset.from_dict(
        {
            'image_paths': numpy.array(image_paths, dtype=str),
            'image_features': numpy.array(image_features, dtype=numpy.float32),
        }
    )
    image_dataset.add_faiss_index(column='image_features', device=accelerator.device.index)

    corpus_names = args.corpus_names.split(",")
    for corpus_name in corpus_names:
        # Load dataset
        corpus_data_dir = os.path.join("data", corpus_name)
        corpus_datasets = load_dataset(corpus_data_dir)

        # Preprocessing the datasets.
        # First we tokenize all the texts.
        column_names = corpus_datasets["test"].column_names

        def preprocess_function(examples):
            inputs = [f"{c} </s> {r}" for c, r in zip(examples['context'], examples['response'])]
            # inputs = examples['context']

            model_inputs = tokenizer(inputs, max_length=tokenizer.max_len_single_sentence, truncation=True,)

            return model_inputs

        with accelerator.main_process_first():
            processed_datasets = corpus_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                desc="Running tokenizer on corpus dataset",
            )

        if args.output_dir is not None:
            vokens_dir = args.output_dir
        else:
            vokens_dir = os.path.join(args.model_path, f'vokens/{corpus_name}')

        if accelerator.is_main_process:
            os.makedirs(vokens_dir, exist_ok=True)

        data_collator = DataCollatorWithPadding(tokenizer)
        for data_type in corpus_datasets:
            dataset = processed_datasets[data_type]
            dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=args.per_device_batch_size)
            dataloader = accelerator.prepare(dataloader)

            if accelerator.is_main_process:
                print(f"Vokenizing {corpus_name} {data_type} data...")
            progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
            retrieved_ids = []
            model.eval()
            accelerator.wait_for_everyone()
            for step, batch in enumerate(dataloader):
                with torch.no_grad():
                    text_feature = accelerator.unwrap_model(model).get_text_features(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                    ).cpu().numpy()

                    scores, retrieved_examples = image_dataset.get_nearest_examples_batch('image_features', text_feature, k=5)
                    retrieved_id = [example['image_paths'] for example in retrieved_examples]

                    if accelerator.num_processes > 1:
                        output = [None for _ in range(accelerator.num_processes)]
                        torch.distributed.all_gather_object(output, retrieved_id)
                        retrieved_id = numpy.concatenate(output).tolist()
                    retrieved_ids.extend(retrieved_id)

                    progress_bar.update(1)

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                with open(os.path.join(vokens_dir, f'{data_type}.json'), 'w') as f:
                    for i, entry in enumerate(corpus_datasets[data_type]):
                        example = {
                            'context': entry['context'],
                            'response': entry['response'],
                            'image_paths': retrieved_ids[i],
                        }
                        f.write(json.dumps(example) + "\n")

            torch.cuda.empty_cache()
