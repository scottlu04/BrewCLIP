#!/bin/bash

# Run trial

#flicker8k
src_dir=/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src
cd ${src_dir}/run
CUDA_VISIBLE_DEVICES=5 python main.py \
--cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/flicker8k/pipeline_gen.yaml' \
--task 'generate_transcrption' \
--trans_data_mode 'train'

# CUDA_VISIBLE_DEVICES=5 python main.py \
# --cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/flicker8k/pipeline_gen.yaml' \
# --task 'generate_transcrption'\
# --trans_data_mode 'dev'

# CUDA_VISIBLE_DEVICES=5 python main.py \
# --cfg_file '/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/config/flicker8k/pipeline_gen.yaml' \
# --task 'generate_transcrption'\
# --trans_data_mode 'test'