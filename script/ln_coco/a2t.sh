#!/bin/bash

# Run trial

src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
cd ${src_dir}/run
CUDA_VISIBLE_DEVICES=0 python main.py \
--train 0 \
--cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/ln_coco/full.yaml' \
--task 'learn' \
--gpus 1 \
--ckpt "/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/run/exp/ln_coco/bifurcated_transformer_prompted_whisper/epoch=15-step=67135-val_recall_mean_1=41.3205.ckpt"

# src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
# cd ${src_dir}/run
# CUDA_VISIBLE_DEVICES=7 python main.py \
# --train 0 \
# --cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/locanarr_coco/maple_1_4.yaml' \
# --task 'pipeline' \
# --gpus 1 \
# --ckpt "/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/run/exp/ln_coco/maple_1_4/epoch=5-step=49999-val_recall_mean_1=38.8615.ckpt"