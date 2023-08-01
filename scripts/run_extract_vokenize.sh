accelerate launch extract_vision_keys.py \
    --model_name_or_path="outputs/contrastive_zrigf" \
    --image_names="open_images,coco" \
    --image_column="image_path" \
    --per_device_batch_size="512" \
    --preprocessing_num_workers="8"

accelerate launch vokenize_corpus.py \
    --model_path="outputs/contrastive_zrigf" \
    --corpus_name="reddit_data,image_chat" \
    --image_names="open_images,coco" \
    --per_device_batch_size="512" \
    --preprocessing_num_workers="8"