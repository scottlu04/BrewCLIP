#!/bin/bash

# Run trial

# #locanarr coco
src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
cd ${src_dir}/run
CUDA_VISIBLE_DEVICES=0 python main.py \
--cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/coco/pipeline_gen.yaml' \
--trans_data_mode 'train' \
--task 'generate_transcrption'
