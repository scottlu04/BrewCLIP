#!/bin/bash

# Run trial

src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
cd ${src_dir}/run
CUDA_VISIBLE_DEVICES=4 python main.py \
--train 0 \
--cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/locanarr_coco/bifurcated.yaml' \
--task 'pipeline' \
--gpus 1 \
#--ckpt "/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/run/lightning_logs/version_23/checkpoints/epoch=2-step=2249.ckpt"