#!/bin/bash

# Run trial

src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
cd ${src_dir}/run
CUDA_VISIBLE_DEVICES=5,6,7 python main.py \
--train 1 \
--cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/locanarr_coco/bifurcated_transformer_prompted.yaml' \
--task 'pipeline' \
--gpus 3 \
--ckpt "/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/run/exp/ln_coco/bifurcated_transformer_prompted/epoch=10-step=46155-val_recall_mean_1=34.8212.ckpt"
# src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
# cd ${src_dir}/run
# CUDA_VISIBLE_DEVICES=7 python main.py \
# --train 0 \
# --cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/locanarr_coco/maple_1_4.yaml' \
# --task 'pipeline' \
# --gpus 1 \
# --ckpt "/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/run/exp/ln_coco/maple_1_4/epoch=5-step=49999-val_recall_mean_1=38.8615.ckpt"