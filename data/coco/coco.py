import json
import os
import datasets

class COCODataset(datasets.GeneratorBasedBuilder):
    """MS COCO dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "image_id": datasets.Value("int64"),
                    "caption_id": datasets.Value("int64"),
                    "caption": datasets.Value("string"),
                    "height": datasets.Value("int64"),
                    "width": datasets.Value("int64"),
                    "file_name": datasets.Value("string"),
                    "coco_url": datasets.Value("string"),
                    "image_path": datasets.Value("string"),
                }
            )
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        data_dir = 'data/coco'
        _DL_URLS = {
            "train": os.path.join(data_dir, "train2017.zip"),
            "val": os.path.join(data_dir, "val2017.zip"),
            "test": os.path.join(data_dir, "test2017.zip"),
            "annotations_trainval": os.path.join(data_dir, "annotations_trainval2017.zip"),
            "image_info_test": os.path.join(data_dir, "image_info_test2017.zip"),
        }
        archive_path = dl_manager.extract(_DL_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "json_path": os.path.join(archive_path["annotations_trainval"], "annotations",
                                              "captions_train2017.json"),
                    "image_dir": os.path.join(archive_path["train"], "train2017"),
                    "split": "train",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "json_path": os.path.join(archive_path["annotations_trainval"], "annotations",
                                              "captions_val2017.json"),
                    "image_dir": os.path.join(archive_path["val"], "val2017"),
                    "split": "valid",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "json_path": os.path.join(archive_path["image_info_test"], "annotations",
                                              "image_info_test2017.json"),
                    "image_dir": os.path.join(archive_path["test"], "test2017"),
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(
        # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
        self, json_path, image_dir, split
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.

        _features = ["image_id", "caption_id", "caption", "height", "width", "file_name", "coco_url", "image_path", "id"]
        features = list(_features)

        if split in "valid":
            split = "val"

        with open(json_path, 'r', encoding='UTF-8') as fp:
            data = json.load(fp)

        # list of dict
        images = data["images"]
        entries = images

        # build a dict of image_id -> image info dict
        d = {image["id"]: image for image in images}

        # list of dict
        if split in ["train", "val"]:
            annotations = data["annotations"]

            # build a dict of image_id ->
            for annotation in annotations:
                _id = annotation["id"]
                image_info = d[annotation["image_id"]]
                annotation.update(image_info)
                annotation["id"] = _id

            entries = annotations

        for id_, entry in enumerate(entries):

            entry = {k: v for k, v in entry.items() if k in features}

            if split == "test":
                entry["image_id"] = entry["id"]
                entry["id"] = -1
                entry["caption"] = -1

            entry["caption_id"] = entry.pop("id")
            entry["image_path"] = os.path.join(image_dir, entry["file_name"])

            entry = {k: entry[k] for k in _features if k in entry}

            yield str((entry["image_id"], entry["caption_id"])), entry
