#!/bin/bash

# Run trial

#flickr 8k
# src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
# cd ${src_dir}/run
# CUDA_VISIBLE_DEVICES=6,7 python main.py \
# --gpus 2 \
# --cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/flicker8k/maple_1_4.yaml' \
# --task 'pipeline' 

# --cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/flicker8k/maple_12_4.yaml' \

src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
cd ${src_dir}/run
CUDA_VISIBLE_DEVICES=4 python main.py \
--gpus 1 \
--cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/flicker8k/maple_1_4.yaml' \
--task 'pipeline' \
--train 0 \
--ckpt '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/run/exp/flickr/maple_1_4/epoch=8-step=8432-val_recall_mean_10=0.0000.ckpt'