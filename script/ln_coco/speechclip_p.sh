#!/bin/bash

# Run trial

src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
cd ${src_dir}/run
CUDA_VISIBLE_DEVICES=2,3,4 python main.py \
--train 1 \
--cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/ln_coco/speechclip_p.yaml' \
--task 'speechclip' \
--gpus 3 \
--ckpt "/media/exx/HDD/zhenyulu/exp/ln_coco/speechclip_p/epoch=2-step=12587-val_recall_mean_1=5.0713.ckpt"

# src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
# cd ${src_dir}/run
# CUDA_VISIBLE_DEVICES=6 python main.py \
# --train 0 \
# --cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/ln_coco/speechclip_p.yaml' \
# --task 'speechclip' \
# --gpus 1 \
# --ckpt '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/run/exp/ln_coco/bifurcated_transformer_prompted_whisper/epoch=3-step=16783-val_recall_mean_1=37.2833.ckpt'

# src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
# cd ${src_dir}/run
# CUDA_VISIBLE_DEVICES=7 python main.py \
# --train 0 \
# --cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/locanarr_coco/maple_1_4.yaml' \
# --task 'pipeline' \
# --gpus 1 \
# --ckpt "/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/run/exp/ln_coco/maple_1_4/epoch=5-step=49999-val_recall_mean_1=38.8615.ckpt"