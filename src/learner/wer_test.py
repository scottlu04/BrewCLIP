import numpy as np
import torch
from easydict import EasyDict as edict
import os, os.path as osp
import yaml
import sys 
import importlib 
import logging
import pickle
from torchmetrics.functional import word_error_rate

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

class Wer_test(object):

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
        #self.model = brewclip(self.cfg)

        # path = self.cfg.pretrained_path
        # checkpoint = torch.load(path)
        # self.brewclip = brewclip(self.cfg).cuda()
        # self.brewclip.load_state_dict(checkpoint["state_dict"])

        self.dataset_name = self.cfg.data.dataset.name
        assert self.dataset_name in dataset_dict
        

    def test(self):
        test_set = getattr(importlib.import_module('.'+ self.dataset_name +'_dataset', package='data'), 
                    dataset_dict[self.dataset_name])(split="test", **self.cfg.data.dataset,)
        test_set.add_transcription_full('aud2trans_test_'+self.dataset_name+'.txt')
        lncoco_test_loader = DataLoader(
                test_set,
                batch_size=1,
                shuffle=False,
                num_workers = 1,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_general,
            )

        
        whisper_lower =[]
        true_lower =[]
        # i = 0
        # for i, x in enumerate(test_set):
        #     whisper_lower.append(x['transcription'])
        #     true_lower.append(x['text'])
        #     print(x)
        #     if i == 1:
        #         break
        with open('/media/exx/HDD/zhenyulu/exp/sim_info.pickle', 'rb') as handle:
            info = pickle.load(handle)
        with open('/media/exx/HDD/zhenyulu/exp/ln_id_path_map.pickle', 'rb') as handle:
            ln_id_path_map = pickle.load(handle)
        with open('/media/exx/HDD/zhenyulu/exp/viz_dict_grad.pickle', 'rb') as handle:
            viz = pickle.load(handle)

        audio_wer_rank ={'id':[],'rank':[],'score':[], 'wer':[] }
        for x in viz:
            for y  in viz[x]['ln'+'_aud_id']:
                audio_wer_rank['id'].append(y)
                audio_wer_rank['rank'].append(info['spk_zs'][1][y])
                audio_wer_rank['score'].append(info['spk_zs'][0][y])
                # sim_score = info[key][0][aud_id]
                # print(info['ln_full'][1][y])    
                # audio_wer_rank.append([])    
        for i in range(len(test_set)):
            print(i)
            
            if i in audio_wer_rank['id']:
                x = test_set[i]
                # whisper_lower.append(x['transcription'])
                # true_lower.append(x['text'])
                wer = float(word_error_rate(x['text'], x['transcription']))
                audio_wer_rank['wer'].append(wer)

        with open('/media/exx/HDD/zhenyulu/exp/wer_rank.pickle', 'wb') as handle:
                pickle.dump(audio_wer_rank, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # print(x)
            # if i == 1:
            #     break
        # overall_WER = word_error_rate(whisper_lower, true_lower)
        # print(overall_WER)
