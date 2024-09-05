#!/bin/bash

src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
cd ${src_dir}/run
CUDA_VISIBLE_DEVICES=7 python main.py \
--train 0 \
--cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/coco/full.yaml' \
--task 'wer_test' \
--gpus 1 \