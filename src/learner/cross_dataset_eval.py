import numpy as np
import torch
from easydict import EasyDict as edict
import os, os.path as osp
import yaml
import sys 
import importlib 
import logging


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader, random_split
if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))
from tqdm import tqdm
sys.path.append('..');
from model import *
from module import asr
from data import collate_general

dataset_dict = {
        'flickr8k' : 'Flickr8kDataset',
        'coco' : 'COCODataset',
        'ln_coco' : 'LN_COCO_Dataset',
        'ln_flickr30k' : 'LN_Flickr30k_Dataset',
        'ravdess' : 'RAVDESS_Dataset',
    }

class Learner(object):

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        seed_everything(self.args.seed)
        
        print("combined cfg:")
        self.cfg.update(edict(vars(self.args)))
        print(self.cfg)
        # if self.cfg.mode == 'pipeline':
        #     self.model = pipeline(self.cfg)
        # elif self.cfg.mode == 'e2e':
        #     self.model = e2e(self.cfg)
        self.model = brewclip(self.cfg)
        self.dataset_name = self.cfg.data.dataset.name
        assert self.dataset_name in dataset_dict
        
    def test(self):
        test_set = getattr(importlib.import_module('.'+ self.dataset_name +'_dataset', package='data'), 
                    dataset_dict[self.dataset_name])(split="test", **self.cfg.data.dataset,)
        test_set.add_transcription_full('aud2trans_test_'+self.dataset_name+'.txt')     
        test_loader = DataLoader(
                test_set,
                batch_size=self.cfg.data.dev_batch_size,
                shuffle=False,
                num_workers=self.cfg.njobs,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_general,
            )
        
        
        trainer = Trainer(
            # callbacks=[
            #     TQDMProgressBar(),
            #     model_checkpoint_val_loss,
            #     model_checkpoint_recall,
            #     *custom_trainer_callbacks,
            # ],
            enable_progress_bar=True,
            gpus=self.cfg.gpus,
           # resume_from_checkpoint = None if self.cfg.ckpt  == "" else self.cfg.ckpt,
            **self.cfg.trainer,
        )
        if self.cfg.ckpt != None:
            trainer.validate(self.model, test_loader, ckpt_path=self.cfg.ckpt)
        else:
            trainer.validate(self.model, test_loader)
    