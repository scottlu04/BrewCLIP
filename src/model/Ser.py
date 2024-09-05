import json
import logging

#logger = logging.getLogger(__name__)

import os
from typing import List, Tuple, Union

import numpy as np
import torch
import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn
from torch.nn import functional as F
import clip
import torchmetrics


import sys
sys.path.append('..');
# from ..base import OrderedNamespace
from module import (
    CustomCLIP,
    FairseqSpeechEncoder_Hubert,
    S3prlSpeechEncoderPlus,
    MLPLayers,
    losses,
    mutualRetrieval,
    whisperx,
    TransformerEncoder
)
from optim import get_scheduler
from module.kw_modules import TransformerModels
# from ..module.speechclip_c_modules import vector_quantizers
# from ..module.speechclip_c_modules.kw_bn import Kw_BatchNorm
# from ..optim import get_scheduler
from util import get_keypadding_mask
from .base_model import BaseLightningModel



class ser(pl.LightningModule):

    def __init__(self, cfg, model):
        super().__init__()
        #self.brewclip = model
        
        #path = "/home/luzhenyu/mmml-audio-retrieval/aud_img_retrieval/src/run/exp/ln_coco/bifurcated_transformer_prompted_whisper/epoch=17-step=75527-val_recall_mean_1=41.3847.ckpt"
        #self.brewclip = model.load_from_checkpoint(path)
        self.cfg = cfg
        path = self.cfg.pretrained_path
        checkpoint = torch.load(path)
        self.brewclip = model
        self.brewclip.load_state_dict(checkpoint["state_dict"])
        if self.cfg.mode == 'linearprob':
            for name, param in self.brewclip.named_parameters():
                    param.requires_grad_(False)
        #self.brewclip.eval()
        #model.eval()
        # checkpoint = torch.load(path, map_location=torch.device('cpu'))
        # self.brewclip.load_state_dict(checkpoint)   
        # #utils.load_state_dict_single(checkpoint['state_dict'],  self.brewclip)
        # self.brewclip.cuda(0)
        self.fc =  nn.Linear(768, 4)
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=4)
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=4)

    def forward(
        self,
        batch,
    ) -> dict:
        #TODO Change back to librosa.laod, to support hubert extraction
        batch['image'] = torch.rand(4, 3, 224, 224).cuda().requires_grad_(requires_grad=False)
        out = self.brewclip(batch)
        #print(out)
        out = self.fc(out['audio_feat'])
        #print(out.size())
        #out = self.f
        # out = {
        #     "id": id,
        #     "image_feat": image_feat,
        #     "audio_feat": final_audio_feat, # actually transcription feat 
        #     "text_feat": text_feat,
        # }
        return {'y_hat':out, 'y': batch['id']}
    
    
    # def validation_epoch_end(self, outputs: list):
    #     """validation_epoch_end

    #     Args:
    #         outputs (list): list of aggregated results
    #     """
    #     # if keywords is in the input, calculate keyword related metrics
    
    #     #MARK
    #     #outputs = outputs['others']
    #     #detach output
    def training_step(self, batch: dict) -> dict:
        out = self.forward(batch)
        #print(out_per_batch['audio_feat'].size())
        loss = nn.functional.cross_entropy(out['y_hat'], out['y'])
        preds = torch.argmax(out['y_hat'], dim=1)
        # acc = self.train_acc(preds, out['y'])
        # self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)
        self.log("train_loss", loss)
        return {"loss": loss}#{"loss_feats": losses, "log_metrics": log_metrics}

    def validation_step(self, batch: dict, batch_idx) -> dict:
        out = self.forward(batch)
        #print(out_per_batch['audio_feat'].size())
        loss = nn.functional.cross_entropy(out['y_hat'], out['y'])
        preds = torch.argmax(out['y_hat'], dim=1)
        acc = self.valid_acc.update(preds, out['y'])
        self.log('val_loss', loss, prog_bar=True)
        # self.log('val_acc', acc, prog_bar=True)
        return {"loss": loss}#{"loss_feats": losses, "log_metrics": log_metrics}
    
    def on_validation_epoch_end(self):
        self.log('valid_acc_epoch', self.valid_acc.compute())
        logging.info(self.valid_acc.compute())
        self.valid_acc.reset()
    def configure_optimizers(self) -> Tuple[list, list]:
        """configure_optimizers

        Returns:
            Tuple[list,list]: (optimizer_list,scheduler_list)
        """
        optimizers = []
        schedulers = []

        my_params = self.getTrainableParams()

        audio_optimizer = getattr(torch.optim, self.cfg.audio_encoder.optim.name)(
            my_params,
            **self.cfg.audio_encoder.optim.args,
        )
        audio_scheduler = get_scheduler(
            optimizer=audio_optimizer,
            **self.cfg.audio_encoder.scheduler,
        )

        optimizers.append(audio_optimizer)
        schedulers.append(
            {
                "scheduler": audio_scheduler,
                "interval": "step",
            }
        )
      #  optimizer = torch.optim.Adam(my_params, lr=0.001)
        return optimizers, schedulers

    def getTrainableParams(self) -> list:
        """getTrainableParams

        return trainable parameter list
        children class should return their additional trainable parameters

        Returns:
            list: list of trainable parameters
        """
        my_params = []
        my_params += self.fc.parameters()
        if self.cfg.mode != 'linearprob':
            my_params += self.brewclip.getTrainableParams()
        return my_params
 