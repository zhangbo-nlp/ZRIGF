# ZRIGF

This is the code for [ZRIGF: An Innovative Multimodal Framework for Zero-Resource Image-Grounded Dialogue Generation.](https://arxiv.org/abs/2308.00400)

## Reference

If you use any source code included in this repo in your work, please cite the following paper.

```
@inproceedings{10.1145/3581783.3611810,
author = {Zhang, Bo and Wang, Jian and Ma, Hui and Xu, Bo and Lin, Hongfei},
title = {ZRIGF: An Innovative Multimodal Framework for Zero-Resource Image-Grounded Dialogue Generation},
year = {2023},
isbn = {9798400701085},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3581783.3611810},
doi = {10.1145/3581783.3611810},
abstract = {Image-grounded dialogue systems benefit greatly from integrating visual information, resulting in high-quality response generation. However, current models struggle to effectively utilize such information in zero-resource scenarios, mainly due to the disparity between image and text modalities. To overcome this challenge, we propose an innovative multimodal framework, called ZRIGF, which assimilates image-grounded information for dialogue generation in zero-resource situations. ZRIGF implements a two-stage learning strategy, comprising contrastive pre-training and generative pre-training. Contrastive pre-training includes a text-image matching module that maps images and texts into a unified encoded vector space, along with a text-assisted masked image modeling module that preserves pre-training visual features and fosters further multimodal feature alignment. Generative pre-training employs a multimodal fusion module and an information transfer module to produce insightful responses based on harmonized multimodal representations. Comprehensive experiments conducted on both text-based and image-grounded dialogue datasets demonstrate ZRIGF's efficacy in generating contextually pertinent and informative responses. Furthermore, we adopt a fully zero-resource scenario in the image-grounded dialogue dataset to demonstrate our framework's robust generalization capabilities in novel domains.},
booktitle = {Proceedings of the 31st ACM International Conference on Multimedia},
pages = {5464–5473},
numpages = {10},
keywords = {image-grounded dialogue, contrastive learning, multimodal fusion, zero resource},
location = {Ottawa ON, Canada},
series = {MM '23}
}
```

## Requirements

* Python 3.10
* Pytorch 2.0
* CUDA 11.8

To install the Python dependencies, run:

```bash
pip install -r requirements.txt
```

To install nlg-eval, run:

```bash
git clone https://github.com/Maluuba/nlg-eval
cd nlg-eval
pip install -e .
```

To make the code work, some files need to be modified:
* `nlg-eval/requirements.txt`: change `gensim~=3.8.3` to `gensim>=3.8.3`
* `nlg-eval/nlgeval/word2vec/evaluate.py`: replace line 40 with the following line:

```python
return vectors[self.m.key_to_index[key]]
```

## Datasets
### Download MS COCO 2017
This example uses COCO dataset (2017) through a custom dataset script, which requires users to manually download the COCO dataset before training.
```bash
cd data/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
```

### Download Open Images
This example uses [Open Images](https://storage.googleapis.com/openimages/web/index.html) images as candidate images for retrieval. To download the images, refer to [here](https://github.com/cvdfoundation/open-images-dataset#download-images-with-bounding-boxes-annotations). You can build the image index with the appropriate size (500,000 in our experiments) as needed.

If you already have Open Images dataset on disk, save them as follows:

```
data
|-- open_images
    |-- images
         |-- 14928b4f367c217e.jpg
         |-- 289d643a8761aa83.jpg
         |-- ......
```

### Download Reddit Conversation
Please download the Reddit data from [here](https://github.com/jokieleung/Maria).

### Download Image-Chat
The Image-Chat dataset can be accessed via [ParlAI](https://github.com/facebookresearch/ParlAI), with -t image_chat.

## Running Codes

Contrastive pre-training:

```bash
bash scripts/run_contrastive_train.sh
```

Extracting the vision features and tokenizing dialogue corpus:

```bash
bash scripts/run_extract_vokenize.sh
```

Generative pre-training:

```bash
bash scripts/run_generative_train.sh
```
