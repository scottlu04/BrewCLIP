#!/bin/bash

# Run trial

#flickr 8k
src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
cd ${src_dir}/run
CUDA_VISIBLE_DEVICES=6 python main.py \
--cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/flicker8k/bifurcated_prompt_cross_modal
.yaml' \
--task 'pipeline' \
--train 0


