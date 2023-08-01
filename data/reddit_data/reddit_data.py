import json

import datasets

SEP = ' </s> '


class RedditData(datasets.GeneratorBasedBuilder):
    """Reddit dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "context": datasets.Value("string"),
                    "response": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "file": "data/reddit_data/train.json",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "file": "data/reddit_data/valid.json",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "file": "data/reddit_data/test.json",
                },
            ),
        ]

    def _generate_examples(self, file):
        with open(file) as f:
            data = json.load(f)
        c = 0
        for episode in data:
            c += 1
            yield c, {
                "context": f'{SEP}'.join(episode['context']),
                "response": episode['response'],
            }
