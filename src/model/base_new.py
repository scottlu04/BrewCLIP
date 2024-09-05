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

import sys
sys.path.append('..');
# from ..base import OrderedNamespace
from module import (
    ClipModel,
    FairseqSpeechEncoder_Hubert,
    MLPLayers,
    losses,
    mutualRetrieval,
)
from optim import get_scheduler
# from ..module.kw_modules import TransformerModels
# from ..module.speechclip_c_modules import vector_quantizers
# from ..module.speechclip_c_modules.kw_bn import Kw_BatchNorm
# from ..optim import get_scheduler
# from ..util import get_keypadding_mask
from .base_model import BaseLightningModel

__all__ = ["Base"]
   # "Tri_Clip"]

"""METRIC_REDUCEFN_MAPPING
define the reduction function for each data type when reducing from multiple GPUs

"""
METRIC_REDUCEFN_MAPPING = {
    torch.Tensor: lambda x: torch.mean(x),
    float: lambda x: x,
    int: lambda x: x,
    str: lambda x: x,
}


class Base(pl.LightningModule):
    """Base Class for SpeechCLIP"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        #self.save_hyperparameters()
        # select audio_encoder type
        

        # the recall to calculate
        self.recall_at = self.cfg.retrieval.recall_at

        # define loss function
        self.criterion = getattr(losses, self.cfg.cl_loss.type)(**self.cfg.cl_loss.args)

    def forward(self, batch: dict) -> tuple:
        
        raise NotImplementedError()
    
    def training_step(self, batch: dict) -> dict:
        out_per_batch = self.forward(batch)
        #losses = self.compute_loss(out_per_batch)
        return {"loss_feats": out_per_batch}#{"loss_feats": losses, "log_metrics": log_metrics}
    
    def training_step_end(self, outputs: dict) -> dict:
        """training_step_end

        Collect results from all GPUs

        Args:
            outputs (dict): output from trainin_step

        Raises:
            NotImplementedError: if the outputs' format collected from GPU(s) is not correct

        Returns:
            dict: loss (return to pytorch lightning for updating params)
        """
        if isinstance(outputs, dict):
            if "loss" in outputs:
                # training_step has already calculated the loss
                # we simply just average the loss on GPU(s)
                return {"loss": torch.mean(outputs["loss"])}
            elif "loss_feats" in outputs:
                losses = self.compute_loss(outputs["loss_feats"])
                return {"loss": losses}
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """validation_step

        Args:
            batch (dict): input data

        Returns:
            dict: output features
        """
        out_per_batch = self.forward(batch)

        # select cascaded or parallel branch's output for contrastive loss calculation
        audio_feat = out_per_batch["audio_feat"] if "audio_feat" in out_per_batch else None
          
        image_feat = out_per_batch["image_feat"] if "image_feat" in out_per_batch else None
        text_feat = out_per_batch["text_feat"] if "text_feat" in out_per_batch else None
        id = out_per_batch["id"]

        # collect features
        return_dict = {
            "id": id,
        }

        if audio_feat is not None:
            return_dict["audio_feat"] = audio_feat
        if image_feat is not None:
            return_dict["image_feat"] = image_feat
        if text_feat is not None:
            return_dict["text_feat"] = text_feat

        #losses = self.compute_loss(out_per_batch)
        #TODO log output
        
        #return {"loss_feats": out_per_batch}#
        return out_per_batch#{"loss_feats": losses, "log_metrics": log_metrics, "others": return_dict}
    

    # def validation_step_end(self, outputs: dict) -> dict:
    #     """validation_step_end

    #     Collect features from all GPU(s) and calculate loss

    #     Args:
    #         outputs (dict): output from GPU(s)

    #     Returns:
    #         dict: features required for validation
    #     """

    #     assert isinstance(outputs, dict)
    #     losses = self.compute_loss(outputs["loss_feats"])

      
    #     return outputs["loss_feats"]


    def validation_epoch_end(self, outputs: list):
        """validation_epoch_end

        Args:
            outputs (list): list of aggregated results
        """
        raise NotImplementedError
        #self.Bi_Retrieval(all_B_feats,all_C_feats,all_B_feats_id,all_C_feats_id,feat_B_name,feat_C_name)
    def Bi_Retrieval(self,all_A_feats,all_B_feats,all_A_feats_id,all_B_feats_id,feat_A_name,feat_B_name ):
        # calculate dot product
        score_per_A = torch.matmul(
            all_A_feats.float().to(self.device),
            all_B_feats.float().T.to(self.device),
        )
        score_per_B = score_per_A.T

        # AI : Audio -> Image, IA: Image -> Audio
        AB_answers = all_A_feats_id
        BA_answers = all_B_feats_id

        recall_results_AB,recall_results_BA = self.reportRetrieval(
            score_per_A=score_per_A,
            score_per_B=score_per_B,
            AB_answers=AB_answers,
            BA_answers=BA_answers,
            metadata = {
            "modality_A_title": feat_A_name,
            "modality_B_title": feat_B_name,
            "modality_A_logAbbr": feat_A_name[0],
            "modality_B_logAbbr": feat_B_name[0],
        },
        )
        return  recall_results_AB,recall_results_BA
    def reportRetrieval(
        self,
        score_per_A: torch.Tensor,
        score_per_B: torch.Tensor,
        AB_answers: torch.Tensor,
        BA_answers: torch.Tensor,
        metadata: dict = {
            "modality_A_title": "audio",
            "modality_B_title": "image",
            "modality_A_logAbbr": "A",
            "modality_B_logAbbr": "I",
        },
    ):
        """reportRetrieval

        Args:
            score_per_A (torch.Tensor): the similarity score per modality A sample
            score_per_B (torch.Tensor): the similarity score per modality B sample
            AB_answers (torch.Tensor): the golden answer (pair ID) for each audio sample
            BA_answers (torch.Tensor): the golden answer (pair ID) for each image sample
            metadata (dict): metadata should include modality the title for A, B and the abbreviation for A and B
        """

        # metadata should include modality the title for A, B and the abbreviation for A and B
        assert "modality_A_title" in metadata
        assert "modality_B_title" in metadata
        assert "modality_A_logAbbr" in metadata
        assert "modality_B_logAbbr" in metadata

        recall_results_AB, recall_results_BA, recall_results_mean = mutualRetrieval(
            score_per_A=score_per_A,
            score_per_B=score_per_B,
            AB_answers=AB_answers,
            BA_answers=BA_answers,
            recall_at=self.recall_at,
            modality_A_title=metadata["modality_A_title"],
            modality_B_title=metadata["modality_B_title"],
        )

        log_AB_abbr = "{}{}".format(
            metadata["modality_A_logAbbr"], metadata["modality_B_logAbbr"]
        )
        log_BA_abbr = "{}{}".format(
            metadata["modality_B_logAbbr"], metadata["modality_A_logAbbr"]
        )

        print(f"val_recall_{log_AB_abbr}", recall_results_AB)
        print(f"val_recall_{log_BA_abbr}", recall_results_BA)
        print("val_recall_mean", recall_results_mean)

        if isinstance(self.logger, WandbLogger):
            # when using wandb
            self.log(f"val_recall_{log_AB_abbr}", recall_results_AB, sync_dist=True)
            self.log(f"val_recall_{log_BA_abbr}", recall_results_BA, sync_dist=True)
            self.log("val_recall_mean", recall_results_mean, sync_dist=True)
        elif isinstance(self.logger, TensorBoardLogger):
            # when using tensorboard
            self.logger.experiment.add_scalars(
                f"val_recall_{log_AB_abbr}", recall_results_AB, self.global_step
            )
            self.logger.experiment.add_scalars(
                f"val_recall_{log_BA_abbr}", recall_results_BA, self.global_step
            )
            self.logger.experiment.add_scalars(
                "val_recall_mean", recall_results_mean, self.global_step
            )
        if self.logger is not None:
            self.log(
                "val_recall_mean_10", recall_results_mean["recall@10"], sync_dist=True
            )
        return recall_results_AB,recall_results_BA
    
    def compute_loss(self, input_feats: dict):
        """compute the loss here

        Args:
            input_feats (dict): the feats required for computing loss
        """
        raise NotImplementedError

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

        return optimizers, schedulers
    def getTrainableParams(self) -> list:
        """getTrainableParams

        return trainable parameter list
        children class should return their additional trainable parameters

        Returns:
            list: list of trainable parameters
        """
        my_params = []
        my_params += list(self.audio_proj_net.parameters())
        if hasattr(self, "audio_encoder"):
            my_params += self.audio_encoder.trainable_params()
            my_params += list(self.criterion.parameters())

        #my_params += self.clip.trainable_params()

        return my_params
    
    def forward_image(self, images: Union[list, torch.Tensor]) -> torch.Tensor:
        """forward_image

        Args:
            images (Union[list, torch.Tensor]): image input

        Raises:
            ValueError: image tensor shape error
            TypeError: image type should be either list or torch.Tensor

        Returns:
            torch.Tensor: image representations
        """
        if isinstance(images, list):
            image_tensor = self.clip.prep_image(images).to(self.device)
        elif isinstance(images, torch.Tensor):
            if images.dim() != 4 or images.shape[1] != 3:
                raise ValueError(f"Incorrect image tensor shape {images.shape}")
            image_tensor = images
        else:
            raise TypeError(f"Unknown image type {type(images)}")

        image_feat = self.clip.encode_image(image_tensor)
        return image_feat

    def forward_text(self, sents: Union[list, torch.Tensor]) -> torch.Tensor:
        if isinstance(sents, list):
            text_tensor = self.clip.prep_text(sents).to(self.device)
        elif isinstance(sents, torch.Tensor):
            sents = sents.squeeze()
            if sents.dim() != 2:
                raise ValueError(f"Incorrect text tensor shape {sents.shape}")
            text_tensor = sents
        else:
            raise TypeError(f"Unknown text type {type(sents)}")
        if hasattr(self.clip, "original2Reduced"):
            # if reduced embedding is used, we need to convert original ids to reduced ids
            for i in range(text_tensor.shape[0]):
                for j in range(text_tensor.shape[1]):
                    text_tensor[i, j] = self.clip.original2Reduced[
                        text_tensor[i, j].item()
                    ]

        text_feat = self.clip.encode_text(text_tensor)
        return text_feat


    def forward_audio(
        self,
        wav: Union[torch.Tensor, list],
        wav_len: Union[torch.Tensor, list] = [],
        return_hidden_states: bool = False,
    ) -> Union[Tuple[Union[torch.Tensor, list], torch.Tensor], torch.Tensor]:
        """Get the representations of audio wav files after passing through the audio encoder

        Args:
            wav (Union[torch.Tensor, list]): wav files
            wav_len (Union[torch.Tensor, list], optional): lengths of each wavform. Defaults to [].
            return_hidden_states (bool, optional): return the hidden representations in the audio encoder. Defaults to False.

        Raises:
            NotImplementedError: if the audio encoder is not implemented in the code

        Returns:
            Union[Tuple[Union[torch.Tensor, list], torch.Tensor], torch.Tensor]: return the representations of waveforms (and also the hidden_states)
        """
        if self.audio_encoder_type in [
            "s3prl_plus",
            "FairseqHubert",
        ]:
            return self.audio_encoder(
                wav, wav_len, return_hidden_states=return_hidden_states
            )
        else:
            raise NotImplementedError("Unknown type:{}".format(self.audio_encoder_type))
