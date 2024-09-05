#!/bin/bash

# Run trial

# src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
# cd ${src_dir}/run
# CUDA_VISIBLE_DEVICES=5,6,7 python main.py \
# --train 1 \
# --cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/coco/full.yaml' \
# --task 'learn' \
# --gpus 3 \
# --ckpt '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/run/exp/coco/bifurcated_transformer_prompted_whisper/epoch=2-step=53132-val_recall_mean_1=46.6232.ckpt'

src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
cd ${src_dir}/run
CUDA_VISIBLE_DEVICES=7 python main.py \
--train 0 \
--cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/coco/full.yaml' \
--task 'pipeline' \
--gpus 1 \
--ckpt "/media/exx/HDD/zhenyulu/exp/coco/full/epoch=5-step=99999-val_recall_mean_1=47.4570.ckpt"