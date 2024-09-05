import numpy as np
import torch
from easydict import EasyDict as edict
import os, os.path as osp
import yaml
import sys 
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader, random_split
if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))
from tqdm import tqdm
sys.path.append('..');
#from model import *
from module import whisperx
from data import CoCoDataset, FlickrDataset, collate_general, LocNarrDataset

class Transcription_generate(object):

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        
        
    def generate(self):

        
        print("combined cfg:")
        self.cfg.update(edict(vars(self.args)))
        print(self.cfg)

        #TODO add resume checkpoint
        self.cfg.ckpt = None

        if self.cfg.data.dataset.name == "flickr":
            if self.args.train:
                tr_set = FlickrDataset(
                    split="train",
                    # load_image=False,
                    # tokenizeText=False,
                    # modalities=["audio", "image", "text"],
                    **self.cfg.data.dataset,
                )
                dv_set = FlickrDataset(
                    split="dev",
                    # load_image=False,
                    # tokenizeText=False,
                    # modalities=["audio", "image", "text"],
                    **self.cfg.data.dataset,
                )
                train_loader = DataLoader(
                    tr_set,
                    batch_size=self.cfg.data.batch_size,
                    shuffle=True,
                    num_workers=self.cfg.njobs,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=collate_general,
                )
                dv_loader = DataLoader(
                    dv_set,
                    batch_size=self.cfg.data.dev_batch_size,
                    shuffle=False,
                    num_workers=self.cfg.njobs,
                    pin_memory=True,
                    drop_last=False,
                    collate_fn=collate_general,
                )
            else:
                #test
                test_set = FlickrDataset(
                    split="test",
                    # load_image=False,
                    # tokenizeText=False,
                    # modalities=["audio", "image", "text"],
                    **self.cfg.data.dataset,
                )
                test_loader = DataLoader(
                test_set,
                batch_size=self.cfg.data.dev_batch_size,
                shuffle=False,
                num_workers=self.cfg.njobs,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_general,
            )
        elif self.cfg.data.dataset.name == "locnarr" :
            #localized narr
            if self.args.train:
                tr_set = LocNarrDataset(
                    split="train",
                    **self.cfg.data.dataset
                )
                dv_set = LocNarrDataset(
                    split="val",
                    **self.cfg.data.dataset
                )
                train_loader = DataLoader(
                    tr_set,
                    batch_size=self.cfg.data.batch_size,
                    shuffle=True,
                    num_workers=self.cfg.njobs,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=collate_general,
                )
                dv_loader = DataLoader(
                    dv_set,
                    batch_size=self.cfg.data.dev_batch_size,
                    shuffle=False,
                    num_workers=self.cfg.njobs,
                    pin_memory=True,
                    drop_last=False,
                    collate_fn=collate_general,
                )
            else:
                # no CoCo test set for locnarr is available, so we just use the validation set
                test_set = LocNarrDataset(
                    split="val",
                    **self.cfg.data.dataset,
                )
                test_loader = DataLoader(
                test_set,
                batch_size=self.cfg.data.dev_batch_size,
                shuffle=False,
                num_workers=self.cfg.njobs,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_general,
                )   
        else:
                raise NotImplementedError

        def has_shared_value(list1, list2):
            for value in list1:
                if value in list2:
                    return True
            return False
        #self.asr.name = 'base.en'
        #self.ASR = asr(self.cfg.ASR.type, self.cfg.ASR.name)
        self.ASR = whisperx.load_model(whisper_arch = self.cfg.ASR.name, device = "cuda", compute_type="float16")
        # file1 = open(self.cfg.data.dataset.dataset_root + "/aud2trans_train.txt","w")
        # bar = tqdm(total=len(train_loader), desc='transcribe steps', dynamic_ncols=False)
        # for ind, x in tqdm(enumerate(train_loader)):
        #     try:    
        #         transcriptions  = self.ASR(x['wav'])
        #         for x in aud_trans:
        #             file1.write(x)
        #             file1.write('\n')
        #      #   print(transcriptions)
        #     except:
        #         transcriptions =['-']
        #         print(x['wav'])
        #     aud_trans  = [x['wav'][i].split("/")[-1][:-4] +'#' + transcriptions[i] for i in range(len(x['wav']))]
        #     #print(aud_trans)
        #     #write to text file
        #     for x in aud_trans:
        #         file1.write(x)
        #         file1.write('\n')
        #     bar.update()
        # file1.close()
        # bar.close()


        file1 = open(self.cfg.data.dataset.dataset_root + "/aud2trans_train.txt","w")
        file1 = open("aud2trans_train.txt","w")
        bar = tqdm(total=len(train_loader), desc='train_data_transcribe steps', dynamic_ncols=False)
        for ind, x in tqdm(enumerate(train_loader)):
            #print(x['wav'].size())
            transcriptions = self.ASR.transcribe(x['wav'].numpy(),batch_size=64,language="en")
            #print(transcriptions)
            aud_trans  = [x['wav_path'][i].split("/")[-1][:-4] +'#' + transcriptions[i] for i in range(len(x['wav']))]
            #print(aud_trans)
            for x in aud_trans:
                file1.write(x)
                file1.write('\n')
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
        
        # file2 = open(self.cfg.data.dataset.dataset_root + "/aud2trans_dev.txt","w")
        # bar = tqdm(total=len(dv_loader), desc='dv_data_transcribe steps', dynamic_ncols=False)
        # for ind, x in tqdm(enumerate(dv_loader)):
        #     #print(x['wav'].size())
        #     transcriptions = self.ASR.transcribe(x['wav'].numpy(),batch_size=64,language="en")
        #     #print(transcriptions)
        #     aud_trans  = [x['wav_path'][i].split("/")[-1][:-4] +'#' + transcriptions[i] for i in range(len(x['wav']))]
        #     #print(aud_trans)
        #     for x in aud_trans:
        #         file2.write(x)
        #         file2.write('\n')
        #     #transcriptions  = self.ASR(x['wav'])
        #     # try:    
        #     #     transcriptions  = self.ASR(x['wav'])
        #     #     aud_trans  = [x['wav'][i].split("/")[-1][:-4] +'#' + transcriptions[i] for i in range(len(x['wav']))]
        #     #     #print(aud_trans)
        #     #     #write to text file
        #     #  #   print(transcriptions)
        #     # except Exception as error:
        #     #     transcriptions =['-']
        #     #     print('failed',error)
        #     #     print(x['wav'])
            
        #     bar.update()
        # file2.close()
        # bar.close()
     
     

    

     
    