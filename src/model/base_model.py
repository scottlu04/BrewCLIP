import abc

import pytorch_lightning as pl
import torch
from torch import nn, optim



class BaseLightningModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

    @abc.abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abc.abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError
