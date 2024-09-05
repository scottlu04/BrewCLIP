#!/bin/bash

src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
cd ${src_dir}/run
CUDA_VISIBLE_DEVICES=1 python main.py \
--train 0 \
--cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/ln_coco/full.yaml' \
--task 'wer_test' \
--gpus 1 \
# --pretrained_path "/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/run/exp/ln_coco/bifurcated_transformer_prompted_whisper/epoch=17-step=75527-val_recall_mean_1=41.3847.ckpt"