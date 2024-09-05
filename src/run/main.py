


import sys
import os, os.path as osp




import argparse
import random
import shutil
import time
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp

import yaml
import importlib
from pprint import pprint
from easydict import EasyDict as edict
from tqdm import tqdm

from copy import deepcopy

#sys.path.append('..')



parser = argparse.ArgumentParser(description='aud_img_retrieval')
parser.add_argument('--train', type=int,default=1, help='train (1) or testval (0) or test (-1).');

parser.add_argument('--cfg_file', type=str, required=True, help='config file to load experimental parameters.');
parser.add_argument('--seed', type=int, default=7122, help='seed');
parser.add_argument('--njobs', type=int, default=4, help='number of workers');
parser.add_argument('--gpus', type=int, default=1, help='number of gpus');
parser.add_argument('--task', type=str, default='e2e', help='train_on_transcription, generate_transcrption, e2e');
parser.add_argument('--transcription_file_name', type=str, default='Flickr8k_aud2trans.txt', help='only for Generate_transcrption mode');
#parser.add_argument('--root_dir', type=str, required=True, help='root directory containing the dataset.');
#parser.add_argument('--log_dir', type=str, required=True, help='directory for logging.');
parser.add_argument('--ckpt', type=str, required=False, 
                    help='directory used for pretraining logs.');
parser.add_argument('--trans_data_mode', type=str, required=False, 
                    help='train or dev or test');
parser.add_argument('--pretrained_path', type=str, required=False, 
                    help='SER experiment only');
args = parser.parse_args()

with open(args.cfg_file, 'rb') as f :
    cfg = edict(yaml.load(f, Loader=yaml.FullLoader)) 
    #cfg = yaml.load(f, Loader=yaml.FullLoader) 

if args.task != 'generate_transcrption':

    if not osp.exists(cfg.trainer.default_root_dir) :
            os.makedirs(cfg.trainer.default_root_dir)
    if args.train == 0:
        log_name = os.path.join(cfg.trainer.default_root_dir, "test.log")
    else:
        log_name = os.path.join(cfg.trainer.default_root_dir, "train.log")


    logging.basicConfig(
                filename=log_name,
                filemode="a",
                format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
                datefmt="%H:%M:%S",
                level=logging.INFO,
            )


SUB_DIR_LEVEL = 1; # level of this subdirectory w.r.t. root of the code
sys.path.append(osp.join(*(['..'] * SUB_DIR_LEVEL)));
from learner import Learner,Transcription_generate, Ser_Learner, Speechclip_learner,Wer_test



if args.task == 'generate_transcrption':
    generator = Transcription_generate(cfg,args)
    generator.generate()
elif args.task == 'ser':
    learner = Ser_Learner(cfg,args)
    if cfg.train:
        learner.train()
    else:
        learner.test()
elif args.task == 'wer_test':
    learner = Wer_test(cfg,args)
    learner.test()
elif args.task == 'speechclip':
    learner = Speechclip_learner(cfg,args)
    if cfg.train:
        learner.train()
    else:
        learner.test()
else:
    learner = Learner(cfg,args)
    if cfg.train:
        learner.train()
    else:
        learner.test()
# elif args.task == 'e2e':
#     learner  = Learner(cfg,args)
#     if cfg.train:
#         learner.train()
#     else:
#         learner.test()
