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
from model import *
from module import asr
from data import CoCoDataset, FlickrDataset, collate_general, LocNarrDataset

class Transcription_learn(object):

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        
        
    def train(self):
        seed_everything(self.args.seed)
        
        print("combined cfg:")
        self.cfg.update(edict(vars(self.args)))
        print(self.cfg)
        self.model = e2e(self.cfg)
        #self.model#.cuda()
        # if self.cfg.mode == 'e2e':
        #     self.model = e2e(self.cfg)
        # elif self.cfg.mode == 'asr':
        #     self.model = ASR_model(self.cfg)
        # else:
        #     raise NotImplementedError

        #TODO add resume checkpoint
        #self.cfg.ckpt = None
        
        if self.cfg.data.dataset.name == "flickr":
            if self.args.train:
                tr_set = FlickrDataset(
                    split="train",
                    # load_image=False,
                    # tokenizeText=False,
                    # modalities=["audio", "image", "text"],
                    **self.cfg.data.dataset,
                )
                tr_set.add_transcription(self.cfg.transcription_file_name)
                dv_set = FlickrDataset(
                    split="dev",
                    # load_image=False,
                    # tokenizeText=False,
                    # modalities=["audio", "image", "text"],
                    **self.cfg.data.dataset,
                )

                dv_set.add_transcription(self.cfg.transcription_file_name)
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
                tr_set.add_transcription_full(self.cfg.transcription_file_name)
               # tr_set.add_transcription(self.cfg.transcription_file_name, 24000)#using first 24000 samples
                train_loader = DataLoader(
                    tr_set,
                    batch_size=self.cfg.data.batch_size,
                    shuffle=True,
                    num_workers=self.cfg.njobs,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=collate_general,
                )

                dv_set = LocNarrDataset(
                    split="val",
                    **self.cfg.data.dataset
                )
               # use 8000 samples for training 
                #dv_set.add_transcription('locanarr_aud2trans_dev.txt', 1000)#using first 24000 samples
                dv_set.add_transcription_full('locanarr_aud2trans_dev.txt')
                dv_loader = DataLoader(
                    dv_set,
                    batch_size=self.cfg.data.dev_batch_size,
                    shuffle=False,
                    num_workers=self.cfg.njobs,
                    pin_memory=True,
                    drop_last=False,
                    collate_fn=collate_general,
                )
                # for i, x in enumerate(dv_loader):
                #     self.model(x)
                #     #rint(x)
                #     break
            else:
                # no CoCo test set for locnarr is available, so we just use the validation set
                test_set = LocNarrDataset(
                    split="val",
                    **self.cfg.data.dataset,
                )
                dv_set.add_transcription(self.cfg.transcription_file_name)
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
        
        from torchmetrics.functional import word_error_rate
        # whisper_lower = [text.lower() for text in whisper_texts]
        # true_lower = [text.lower() for text in true_texts]
        # WERs = [float(word_error_rate(whisper_lower[i], true_lower[i])) for i in range(len(whisper_lower))]
        # overall_WER = word_error_rate(whisper_lower, true_lower)
        whisper_lower =[]
        true_lower =[]
        for  x in dv_set:
            whisper_lower.append(x['transcription'])
            #true_lower.append(x['text'])
            #print(whisper_lower)
            print(x['image'])
            print(x['transcription'])
            # print(true_lower)
        overall_WER = word_error_rate(whisper_lower, true_lower)
        print(overall_WER)
       # exit()
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
        # trainer.validate(self.model, dv_loader, ckpt_path=self.cfg.ckpt, verbose=True)
        #trainer.validate(self.model, dv_loader, verbose=True)
        #exit()
        if self.args.train:
            # trainer.validate(model, tr_loader, ckpt_path=self.args.ckpt, verbose=True)
            trainer.fit(self.model, train_loader, dv_loader, ckpt_path=self.cfg.ckpt)
        if self.args.eval:
            trainer.validate(model, dv_loader, ckpt_path=config.ckpt, verbose=True)
        if self.args.test:
            # test_func = getattr(model, "test_step", None)
            # if callable(test_func):
            #     # test utility is implemented and callable
            #     trainer.test(model, test_loader, ckpt_path=config.ckpt)
            # else:
            #     # use validate function instead.
            trainer.validate(model, test_loader, ckpt_path=config.ckpt)

    
    