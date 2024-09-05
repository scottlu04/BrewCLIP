#!/bin/bash

# Run trial

#flickr 8k
# src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
# cd ${src_dir}/run
# CUDA_VISIBLE_DEVICES=6,7 python main.py \
# --gpus 2 \
# --cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/flicker8k/bifurcated.yaml' \
# --task 'pipeline' 

# --cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/flicker8k/maple_12_4.yaml' \
#export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
cd ${src_dir}/run
CUDA_VISIBLE_DEVICES=6,7 python main.py \
--gpus 2 \
--cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/flicker8k/bifurcated.yaml' \
--task 'pipeline' \
--train 0 \
--ckpt '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/run/exp/flickr/bifurcated/epoch=14-step=14054-val_recall_mean_1=65.5200.ckpt'