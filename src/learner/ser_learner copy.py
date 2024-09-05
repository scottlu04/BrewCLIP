import numpy as np
import torch
from easydict import EasyDict as edict
import os, os.path as osp
import yaml
import sys 
import importlib 
import logging
import torchmetrics

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
       'ravdess' : 'RAVDESS_Dataset',
    }

class Ser_Learner(object):

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
        self.brewclip = brewclip(self.cfg)
        self.ser_model = ser(self.cfg, self.brewclip).cuda()
        self.dataset_name = self.cfg.data.dataset.name
        assert self.dataset_name in dataset_dict
        
    def train(self):
        if self.args.train:
            tr_set = getattr(importlib.import_module('.'+ self.dataset_name +'_dataset', package='data'), 
                    dataset_dict[self.dataset_name])(split="train", **self.cfg.data.dataset,)
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
            dv_loader = DataLoader(
                        dv_set,
                        batch_size=self.cfg.data.dev_batch_size,
                        shuffle=False,
                        num_workers=self.cfg.njobs,
                        pin_memory=True,
                        drop_last=False,
                        collate_fn=collate_general,
                    )
            
        #self.model.cuda(self.args.gpu)
        def tensor_dict_to_cuda(tdict: dict, gpu: int) :
            for k in tdict :
                if isinstance(tdict[k], dict) :
                    tensor_dict_to_cuda(tdict[k], gpu)
                    continue
                
                if torch.is_tensor(tdict[k]) :
                    tdict[k] = tdict[k].cuda(gpu, non_blocking=True) 
        for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
            save_model = (epoch + 1 == self.arg.num_epoch) and (epoch + 1 > self.arg.save_epoch)

            self.train(epoch, save_model=save_model)

            # if epoch > 80:
            self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])  
        # bar = tqdm(total=len(train_loader), desc='train_data_transcribe steps', dynamic_ncols=False)
        for ind, x in enumerate(train_loader):
            tensor_dict_to_cuda(x,0)
            out = self.ser_model(x)
        # bar.close()
def train_epoch(self, tbar, epoch, train_logger, current_t_index) :

 
        # Class to save epoch metrics 
        acc_meter = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).cuda(self.args.gpu)

        n_batches = len(self.train_loader)

        # set to train mode
        self.model.train()
        tbar.reset(total=n_batches)
        tbar.refresh()

    

        iter_loader = iter(self.train_loader)
        bi = 1
        while bi <= n_batches :
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data.pts, data.label
        
            # Map target
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]

            output = self.model(pts)[:, :self.valid_out_dim]

            loss_tensors = []

            for lname in self.criteria :
                lfunc = self.criteria[lname].func
                lweight = self.criteria[lname].weight
                lval = lfunc(output, target)
                losses[lname].update(lval.item(), output.size(0) )
                loss_tensors.append(lweight * lval)

            loss = sum(loss_tensors)

            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()

            train_acc= acc_meter(output, target) * 100

            if self.args.gpu == 0 :           
                tbar.update()
                tbar.set_postfix({
                        'it': bi,
                        'loss': loss.item(), 
                        'train_acc': train_acc.item(),
                })
                tbar.refresh()

            bi += 1
        acc_all = acc_meter.compute() * 100
        acc_meter.reset()
        # if self.args.gpu == 0 :
        #     acc_all = acc_meter.compute() * 100
        #     # hyperparam update
        #     train_logger.update(
        #         {'learning_rate': self.optimizer.param_groups[0]['lr']},
        #         step=epoch, prefix="stepwise")

        #     # loss update
        #     train_logger.update(
        #         { ltype: lmeter.avg for ltype, lmeter in losses.items() },
        #         step=epoch, prefix="loss")

        #     # measures update
        #     train_logger.update({
        #         'mean': acc_all,
        #         }, step=epoch, prefix="acc" )                

        #     acc_meter.reset()
        #     train_logger.flush()  
       

    @torch.no_grad()
    def validate_epoch(self, vbar, epoch, val_logger) :
   
        losses = edict({
            name: utils.AverageMeter() for name in self.criteria
        })
        
        # Class to save epoch metrics
        acc_meter = torchmetrics.Accuracy(task='multiclass', num_classes=self.valid_out_dim).cuda(self.args.gpu)

        # set to eval mode
        self.model.eval()

        if self.args.gpu == 0 :
            vbar.reset(total=len(self.val_loader))
            vbar.refresh()

        n_batches = len(self.val_loader)
        iter_loader = iter(self.val_loader)
        bi = 1

        while bi <= n_batches :
            data = next(iter_loader)
            # transfer data to gpu
            utils.tensor_dict_to_cuda(data, self.args.gpu)

            pts, target = data.pts, data.label
            # Map target
            for i, target_class in enumerate(target):
                target[i] = self.cfg.class_mapping[str(target_class.item())]

            output = self.model(pts)[:, :self.valid_out_dim]

            loss_tensors = []
            for lname in self.criteria :
                lfunc = self.criteria[lname].func
                lweight = self.criteria[lname].weight
                lval = lfunc(output, target)
                losses[lname].update(lval.item(), output.size(0))
                loss_tensors.append(lweight * lval)

            loss = sum(loss_tensors)

            val_acc = acc_meter(output, target) * 100

            if self.args.gpu == 0 :
                vbar.update()
                vbar.set_postfix({
                    'it': bi,
                    'loss': loss.item(), 
                    'val_acc': val_acc.item(),
                })                
                vbar.refresh()

            bi += 1   

        if self.args.gpu == 0 :
            acc_all = acc_meter.compute() * 100
            
            # loss update
            val_logger.update(
                { ltype: lmeter.avg for ltype, lmeter in losses.items() },
                step=epoch, prefix="loss")

            # measures update
            val_logger.update({
                'mean': acc_all,
                }, step=epoch, prefix="acc" )                

            acc_meter.reset()               
            val_logger.flush() 

            return_values = { ltype: lmeter.avg for ltype, lmeter in losses.items() }
            return_values['acc'] = acc_all
            
            return return_values

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
    