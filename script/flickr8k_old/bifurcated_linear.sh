#!/bin/bash

# Run trial

#flickr 8k

#train 
src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
cd ${src_dir}/run
CUDA_VISIBLE_DEVICES=6,7 python main.py \
--gpus 2 \
--cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/flicker8k/bifurcated_linear.yaml' \
--task 'pipeline' \
#--train 1 \
#--ckpt '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/run/exp/flickr/bifurcated/epoch=14-step=14054-val_recall_mean_1=65.5200.ckpt'

#test
# src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
# cd ${src_dir}/run
# CUDA_VISIBLE_DEVICES=6,7 python main.py \
# --gpus 2 \
# --cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/flicker8k/bifurcated_linear.yaml' \
# --task 'pipeline' \
# --train 0 \
# #--ckpt '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/run/exp/flickr/bifurcated/epoch=14-step=14054-val_recall_mean_1=65.5200.ckpt'