#!/bin/bash

# Run trial

src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
cd ${src_dir}/run
CUDA_VISIBLE_DEVICES=1 python main.py \
--train 0 \
--cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/ln_coco/pipeline_only_zs_gt.yaml' \
--task 'learn' \
--gpus 1 \