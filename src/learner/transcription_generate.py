import numpy as np
import torch
from easydict import EasyDict as edict
import os, os.path as osp
import yaml
import sys
import importlib 
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader, random_split
if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))
from tqdm import tqdm
sys.path.append('..');

# dir_path = os.path.dirname(os.path.realpath(__file__))
# parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
# sys.path.insert(0, parent_dir_path)

#from model import *
from module import whisperx
from data import collate_general

dataset_dict = {
        'flickr8k' : 'Flickr8kDataset',
        'coco' : 'COCODataset',
        'ln_coco' : 'LN_COCO_Dataset',
        'ln_flickr30k' : 'LN_Flickr30k_Dataset',
    }
class Transcription_generate(object):

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.ASR = whisperx.load_model(whisper_arch = self.cfg.ASR.name, device = "cuda", compute_type="float16")
        #self.dataset_mode = ['train','dev','test']
        self.trans_data_mode = self.args.trans_data_mode
        #self.dataset_mode = ['train']

    def generate(self):

        
        print("combined cfg:")
        self.cfg.update(edict(vars(self.args)))
        print(self.cfg)
        seed_everything(self.args.seed)
        #TODO add resume checkpoint
        #self.cfg.ckpt = None
        dataset_name = self.cfg.data.dataset.name
        assert dataset_name in dataset_dict
        if 'train' == self.trans_data_mode:
            tr_set = getattr(importlib.import_module('.'+ dataset_name +'_dataset', package='data'), 
                    dataset_dict[dataset_name])(split="train", **self.cfg.data.dataset)
            train_loader = DataLoader(
                        tr_set,
                        batch_size=self.cfg.data.batch_size,
                        shuffle=True,
                        num_workers=self.cfg.njobs,
                        pin_memory=True,
                        drop_last=True,
                        collate_fn=collate_general,
                    )
            
            file1 = open(self.cfg.data.dataset.dataset_root + "/aud2trans_train_"+dataset_name +".txt","w")
            #file1 = open("aud2trans_train.txt","w")
            bar = tqdm(total=len(train_loader), desc='train_data_transcribe steps', dynamic_ncols=False)
            for ind, x in tqdm(enumerate(train_loader)):
                #print(x['wav'].size())
                transcriptions = self.ASR.transcribe(x['wav'].numpy(),batch_size=64,language="en")
                # print(transcriptions)
                # print(x['wav_path'])
                aud_trans  = [x['wav_path'][i].split("/")[-1][:-4] +'#' + transcriptions[i] for i in range(len(x['wav']))]
                for z in aud_trans:
                    file1.write(z)
                    file1.write('\n')
                # if ind == 514:
                #     break
                #transcriptions  = self.ASR(x['wav'])
                # try:    
                #     transcriptions  = self.ASR(x['wav'])
                #     aud_trans  = [x['wav'][i].split("/")[-1][:-4] +'#' + transcriptions[i] for i in range(len(x['wav']))]
                #     #print(aud_trans)
                #     #write to text file
                #  #   print(transcriptions)
                # except Exception as error:
                #     transcriptions =['-']
                #     print('failed',error)
                #     print(x['wav'])
                
                bar.update()
            file1.close()
            bar.close()
        if 'dev' == self.trans_data_mode:
            dv_set = getattr(importlib.import_module('.'+ dataset_name +'_dataset', package='data'), 
                    dataset_dict[dataset_name])(split="dev", **self.cfg.data.dataset,)
            dv_loader = DataLoader(
                        dv_set,
                        batch_size=self.cfg.data.dev_batch_size,
                        shuffle=False,
                        num_workers=self.cfg.njobs,
                        pin_memory=True,
                        drop_last=False,
                        collate_fn=collate_general,
                    )

            #file2 = open(self.cfg.data.dataset.dataset_root + "/aud2trans_dev.txt","w")
            bar = tqdm(total=len(dv_loader), desc='dv_data_transcribe steps', dynamic_ncols=False)
            for ind, x in tqdm(enumerate(dv_loader)):
                #print(x['wav'].size())
                transcriptions = self.ASR.transcribe(x['wav'].numpy(),batch_size=64,language="en")
                #print(transcriptions)
                aud_trans  = [x['wav_path'][i].split("/")[-1][:-4] +'#' + transcriptions[i] for i in range(len(x['wav']))]
                #print(aud_trans)
                # for x in aud_trans:
                #     file2.write(x)
                #     file2.write('\n')      
                bar.update()
            #file2.close()
            bar.close()

        if 'test' in self.trans_data_mode:
            test_set = getattr(importlib.import_module('.'+ dataset_name +'_dataset', package='data'), 
                    dataset_dict[dataset_name])(split="test", **self.cfg.data.dataset,)
            test_loader = DataLoader(
                    test_set,
                    batch_size=self.cfg.data.dev_batch_size,
                    shuffle=False,
                    num_workers=self.cfg.njobs,
                    pin_memory=True,
                    drop_last=False,
                    collate_fn=collate_general,
                )
            
            file3 = open(self.cfg.data.dataset.dataset_root + "/aud2trans_test_"+dataset_name +".txt","w")
            bar = tqdm(total=len(test_loader), desc='test_data_transcribe steps', dynamic_ncols=False)
            for ind, x in tqdm(enumerate(test_loader)):
                #print(x['wav'].size())
                transcriptions = self.ASR.transcribe(x['wav'].numpy(),batch_size=64,language="en")
                #print(transcriptions)
                aud_trans  = [x['wav_path'][i].split("/")[-1][:-4] +'#' + transcriptions[i] for i in range(len(x['wav']))]
                #print(aud_trans)
                for x in aud_trans:
                    file3.write(x)
                    file3.write('\n')
    
                bar.update()
            file3.close()
            bar.close()
        


        
 
     
     

    

     
    