import os

import datasets


class OpenImages(datasets.GeneratorBasedBuilder):
    """Open Images dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "image_id": datasets.Value("string"),
                    "image_path": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "files": dl_manager.iter_files(['data/open_images/images']),
                },
            ),
        ]

    def _generate_examples(self, files):
        for i, path in enumerate(files):
            file_name = os.path.basename(path)
            if file_name.endswith(".jpg"):
                yield i, {
                    "image_id": file_name.replace('.jpg', ''),
                    "image_path": path,
                }
