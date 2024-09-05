#!/bin/bash

# Run trial

#flickr 8k
# src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
# cd ${src_dir}/run
# CUDA_VISIBLE_DEVICES=5 python main.py \
# --cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/flicker8k/ASR_model.yaml' \
# --task 'train_on_transcription'


# #locanarr coco
src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
cd ${src_dir}/run
CUDA_VISIBLE_DEVICES=5 python main.py \
--cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/locanarr_coco/ASR_model.yaml' \
--task 'train_on_transcription' \
--gpus 1 \
--transcription_file_name 'locanarr_aud2trans_train.txt' \
#--ckpt "/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/run/lightning_logs/version_23/checkpoints/epoch=2-step=2249.ckpt"