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
        
    def train(self):
        if self.args.train:
            tr_set = getattr(importlib.import_module('.'+ self.dataset_name +'_dataset', package='data'), 
                    dataset_dict[self.dataset_name])(split="train", **self.cfg.data.dataset,)
    
            tr_set.add_transcription_full('aud2trans_train_'+self.dataset_name+'.txt')
            train_loader = DataLoader(
                        tr_set,
                        batch_size=self.cfg.data.batch_size,
                        shuffle=True,
                        num_workers=self.cfg.njobs,
                        pin_memory=True,
                        drop_last=True,
                        collate_fn=collate_general,
                    )
            
            # dv_set = getattr(importlib.import_module('.'+ self.dataset_name +'_dataset', package='data'), 
            #         dataset_dict[self.dataset_name])(split="dev", **self.cfg.data.dataset,)
            dv_set = getattr(importlib.import_module('.'+ self.dataset_name +'_dataset', package='data'), 
                    dataset_dict[self.dataset_name])(split="test", **self.cfg.data.dataset,)
            
            #dv_set.add_transcription_full('aud2trans_dev.txt')

            dv_set.add_transcription_full('aud2trans_test_'+self.dataset_name+'.txt')
            dv_loader = DataLoader(
                        dv_set,
                        batch_size=self.cfg.data.dev_batch_size,
                        shuffle=False,
                        num_workers=self.cfg.njobs,
                        pin_memory=True,
                        drop_last=False,
                        collate_fn=collate_general,
                    )
            
        # bar = tqdm(total=len(train_loader), desc='train_data_transcribe steps', dynamic_ncols=False)
        # for ind, x in tqdm(enumerate(train_loader)):
        #     #print(x['wav'].size())
        #     #print(x)
        #     continue
       
        # bar.close()
        # exit()
        model_checkpoint_recall = ModelCheckpoint(
            dirpath=self.cfg.trainer.default_root_dir,
            filename="{epoch}-{step}-{val_recall_mean_1:.4f}",
            monitor="val_recall_mean_1",
            save_top_k=3,
            mode="max",
            every_n_epochs=1,
        )
        trainer = Trainer(
            callbacks=[
                TQDMProgressBar(),
                # model_checkpoint_val_loss,
                model_checkpoint_recall,
                # *custom_trainer_callbacks,
            ],
            enable_progress_bar=True,
            gpus=self.cfg.gpus,
           resume_from_checkpoint = None if self.cfg.ckpt  == "" else self.cfg.ckpt,
            **self.cfg.trainer,
        )
        # trainer.validate(self.model, dv_loader, ckpt_path=self.cfg.ckpt, verbose=True)
        #trainer.validate(self.model, dv_loader, verbose=True)
        #exit()
        if self.args.train:
            # trainer.validate(model, tr_loader, ckpt_path=self.args.ckpt, verbose=True)
            trainer.fit(self.model, train_loader, dv_loader, ckpt_path=self.cfg.ckpt)
        # if self.args.eval:
        #     trainer.validate(model, dv_loader, ckpt_path=config.ckpt, verbose=True)
        # if self.args.test:
        #     # test_func = getattr(model, "test_step", None)
        #     # if callable(test_func):
        #     #     # test utility is implemented and callable
        #     #     trainer.test(model, test_loader, ckpt_path=config.ckpt)
        #     # else:
        #     #     # use validate function instead.
        #     trainer.validate(model, test_loader, ckpt_path=config.ckpt)
    def test(self):
        test_set = getattr(importlib.import_module('.'+ self.dataset_name +'_dataset', package='data'), 
                    dataset_dict[self.dataset_name])(split="test", **self.cfg.data.dataset,)
        test_set.add_transcription_full('aud2trans_test_'+self.dataset_name+'.txt')
        # image_feat_list = []
        # text_feat_list = []
        # whisper_text_feat_list = []
        # ids_seen = set()
        # whisper_texts = []
        # true_texts = []
        # img_to_text_idx = defaultdict(lambda : [])
        # text_to_img_idx = defaultdict(lambda : [])
        # id_to_img_idx = dict()
        # for idx, sample in tqdm(enumerate(test_loader)): #could also tqdm over dev_set
        #     text = sample['text']
        #     img = sample['image']
        #     wav = sample['wav']
        #     whisper_text = wav_to_text(wav, whisper_model, options)
        #     text_features = text_to_embed(text, model)
        #     whisper_text_features = text_to_embed(whisper_text, model)
        #     image_features = img_to_embed(img, model, preprocess)
            
            
        #     # text_feat_list  += [text_features[0].tolist()]
        #     # whisper_text_feat_list += [whisper_text_features[0].tolist()]
        #     # whisper_texts += [whisper_text]
        #     # true_texts += [text]
            
        #     if sample['id'] not in ids_seen:
        #         image_feat_list += image_features.tolist()
        #         id_to_img_idx[sample['id']] = len(image_feat_list)-1
        #         ids_seen.add(sample['id'])
            
        #     cur_image_idx = id_to_img_idx[sample['id']]
        #     cur_text_idx = len(true_texts)-1
            
        #     img_to_text_idx[cur_text_idx] += [cur_image_idx]
        #     text_to_img_idx[cur_image_idx] += [cur_text_idx]
        #     if idx == coco_samples_to_eval-1:
        #         break
        
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
    