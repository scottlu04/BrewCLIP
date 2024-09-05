#!/bin/bash

# Run trial

#flicker8k

# src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
# cd ${src_dir}/run
# CUDA_VISIBLE_DEVICES=0,1 python main.py \
# --cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/flicker8k/base.yaml' \
# --gpus 2



#locanarr 
#with clip text encoder
src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
cd ${src_dir}/run
CUDA_VISIBLE_DEVICES=4,5 python main.py \
--cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/locanarr_coco/e2e.yaml' \
--transcription_file_name 'locanarr_aud2trans_train.txt' \
--task 'pipeline' \
--train 1 \
--gpus 2