import json
import logging
import clip
logger = logging.getLogger(__name__)

import os
from typing import List, Tuple, Union

import numpy as np
import torch
import tqdm
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import nn
from torch.nn import functional as F


from ..base import OrderedNamespace
from ..module import (
    ClipModel,
    FairseqSpeechEncoder_Hubert,
    MLPLayers,
    S3prlSpeechEncoderPlus,
    losses,
    mutualRetrieval,
)
from ..module.kw_modules import TransformerModels
from ..module.speechclip_c_modules import vector_quantizers
from ..module.speechclip_c_modules.kw_bn import Kw_BatchNorm
from ..optim import get_scheduler
from ..util import get_keypadding_mask
from .base_model import BaseLightningModel
from .Base import KWClipBase

__all__ = [
    "Tri_Clip"]


class Tri_Clip(KWClipBase):
    """KWClip_GeneralTransformer
    Main class for SpeechCLIP
    """

    def __init__(self, config: OrderedNamespace) -> None:
        """init

        Args:
            config (OrderedNamespace): _description_
        """
        super().__init__(config)


        self.parallel_branch = None
        if self.config.model_settings.parallel_objective_weight > 0:
            logger.info("Create Parallel Branch")
            self.parallel_branch = KW_ParallelBranch(
                config=self.config,
                audio_dim=self.audio_embd_dim,
                out_dim=self.subword_embd_dim,
            )

        #self.audio_proj_net = nn.Linear(768, 512)
            

        # projection network after CLIP image encoder
        self.img_enc_proj_net = None
        image_encoder_projection = self.config.model_settings.get(
            "image_encoder_projection", None
        )
        if image_encoder_projection is not None:
            logger.info(
                f"image_encoder_projection dims:{image_encoder_projection.dimensions} droupout:{image_encoder_projection.dropout}"
            )
            self.img_enc_proj_net = MLPLayers(
                units=image_encoder_projection.dimensions,
                dropout=image_encoder_projection.dropout,
            )


        # projection network after CLIP text encoder
        self.text_enc_proj_net = None
        text_encoder_projection = self.config.model_settings.get(
            "text_encoder_projection", None
        )
        if text_encoder_projection is not None:
            logger.info(
                f"text_encoder_projection dims:{text_encoder_projection.dimensions} droupout:{text_encoder_projection.dropout}"
            )
            self.text_enc_proj_net = MLPLayers(
                units=text_encoder_projection.dimensions,
                dropout=text_encoder_projection.dropout,
            )


        # projection network after parallel branch
        self.p_branch_proj_net = None
        parallel_branch_projection = self.config.model_settings.get(
            "parallel_branch_projection", None
        )
        if parallel_branch_projection is not None:
            logger.info(
                f"parallel_branch_projection dims:{parallel_branch_projection.dimensions} droupout:{parallel_branch_projection.dropout}"
            )
            self.p_branch_proj_net = MLPLayers(
                units=parallel_branch_projection.dimensions,
                dropout=parallel_branch_projection.dropout,
            )




    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """validation_step

        Args:
            batch (dict): input data

        Returns:
            dict: output features
        """
        losses, log_metrics, others = self.forward(batch)

        # select cascaded or parallel branch's output for contrastive loss calculation
        audio_feat = others["audio_feat"] if "audio_feat" in others else None
          
        image_feat = others["image_feat"] if "image_feat" in others else None
        text_feat = others["text_feat"] if "text_feat" in others else None
        id = others["id"]

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

        if "keywords" in others and others["keywords"] is not None:
            keywords = others["keywords"]
            return_dict["keywords"] = keywords
            return_dict["gold_text"] = batch["text"]

        return {"loss_feats": losses, "log_metrics": log_metrics, "others": return_dict}

    def validation_epoch_end(self, outputs: list):
        """validation_epoch_end

        Args:
            outputs (list): list of aggregated results
        """
        # if keywords is in the input, calculate keyword related metrics
    
        #MARK
        modality_list = self.config.model_settings.modality 
        assert len(modality_list) == 3
        #B image  A text
        all_ids = torch.cat([x["id"] for x in outputs], dim=0)
        feat_dict = []
        for mod in modality_list:
            feat_name = mod+"_feat"
            all_feats = torch.cat([x[feat_name] for x in outputs], dim=0)
            if mod =='image':
                id_feat_pairs = {_id.item(): _x for _id, _x in zip(all_ids, all_feats)}
                all_feats = torch.stack([x for _, x in id_feat_pairs.items()], dim=0)
                all_feats_id = torch.LongTensor(list(id_feat_pairs.keys()))
            else:
                all_feats_id = all_ids
            feat_dict.append([mod,all_feats, all_feats_id])
            print("Total #{} {}".format(len(all_feats),mod))
    
        all_A_feats = feat_dict[0][1]
        all_A_feats_id = feat_dict[0][2]
        feat_A_name = feat_dict[0][0]
        all_B_feats = feat_dict[1][1]
        all_B_feats_id = feat_dict[1][2]
        feat_B_name = feat_dict[1][0]
        all_C_feats = feat_dict[2][1]
        all_C_feats_id = feat_dict[2][2]
        feat_C_name = feat_dict[2][0]
        recall_results_AB,recall_results_BA = self.bi_Retrieval(all_A_feats,all_B_feats,all_A_feats_id,all_B_feats_id,feat_A_name,feat_B_name)
        recall_results_AC,recall_results_CA =self.bi_Retrieval(all_A_feats,all_C_feats,all_A_feats_id,all_C_feats_id,feat_A_name,feat_C_name)
        self.bi_Retrieval(all_B_feats,all_C_feats,all_B_feats_id,all_C_feats_id,feat_B_name,feat_C_name)
        recall_results_AB = list(recall_results_AB.values())
        recall_results_BA = list(recall_results_BA.values())
        recall_results_AC = list(recall_results_AC.values())
        recall_results_CA = list(recall_results_CA.values())
        outfile = f"& {recall_results_AB[0] :.1f} & {recall_results_AB[1] :.1f} & {recall_results_AB[2] :.1f} "
        outfile += f"& {recall_results_BA[0] :.1f} & {recall_results_BA[1] :.1f} & {recall_results_BA[2] :.1f} "
        outfile += f"& {recall_results_AC[0] :.1f} & {recall_results_AC[1] :.1f} & {recall_results_AC[2] :.1f} "
        outfile += f"& {recall_results_CA[0] :.1f} & {recall_results_CA[1] :.1f} & {recall_results_CA[2] :.1f}"
        
        print(outfile)
        #self.bi_Retrieval(all_B_feats,all_C_feats,all_B_feats_id,all_C_feats_id,feat_B_name,feat_C_name)


    def bi_Retrieval(self,all_A_feats,all_B_feats,all_A_feats_id,all_B_feats_id,feat_A_name,feat_B_name ):
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
    def getTrainableParams(self) -> list:
        """getTrainableParams

        Returns:
            list: list of trainable params in this class
        """
        _params = super().getTrainableParams()

        if self.parallel_branch is not None:
            logger.info("Add parallel_branch parameters")
            _params += list(self.parallel_branch.parameters())
        else:
        #TODO
            logger.info("Add audio linear parameters")
            _params += list(self.audio_proj_net.parameters())

        if self.img_enc_proj_net is not None:
            logger.info("Add img_enc_proj_net parameters")
            _params += list(self.img_enc_proj_net.parameters())

        if self.text_enc_proj_net is not None:
            logger.info("Add text_enc_proj_net parameters")
            _params += list(self.text_enc_proj_net.parameters())

        if self.p_branch_proj_net is not None:
            logger.info("Add parallel_branch_projection parameters")
            _params += list(self.p_branch_proj_net.parameters())

        return _params

    def compute_loss(self, input_feats: dict):
        """compute the loss here

        Args:
            input_feats (dict): the feats required for computing loss
        """
        assert isinstance(input_feats, dict)
 
        modality_list = self.config.model_settings.modality
        
        assert len(modality_list) == 3
        # if 'audio' in modality_list:
        #     assert "audio_feat" in input_feats 
        #     audio_feat = input_feats["audio_feat"].float()
        # if 'image' in modality_list:
        #     assert "image_feat" in input_feats
        #     image_feat = input_feats["image_feat"].float()
        # if 'text' in modality_list:
        #     assert "text_feat" in input_feats
        #     text_feat = input_feats["text_feat"].float()

        feat_A_name = modality_list[0]+"_feat"
        assert feat_A_name in input_feats 
        feat_A = input_feats[feat_A_name].float()
        
        feat_B_name = modality_list[1]+"_feat"
        assert feat_B_name in input_feats 
        feat_B = input_feats[feat_B_name].float()

        feat_C_name = modality_list[2]+"_feat"
        assert feat_C_name in input_feats 
        feat_C = input_feats[feat_C_name].float()

        assert "id" in input_feats
        id = input_feats["id"]

        losses = {"loss": 0}
        cl_loss_AB = "c_cl_loss_"+feat_A_name[0]+feat_B_name[0]
        losses[cl_loss_AB] = self.criterion(
            feat_A=feat_A,
            feat_B=feat_B,
            index=id,
        )
        cl_loss_AC = "c_cl_loss_"+feat_A_name[0]+feat_C_name[0]
        losses[cl_loss_AC] = self.criterion(
            feat_A=feat_A,
            feat_B=feat_C,
            index=id,
        )
        cl_loss_BC = "c_cl_loss_"+feat_B_name[0]+feat_C_name[0]
        losses[cl_loss_BC] = self.criterion(
            feat_A=feat_B,
            feat_B=feat_C,
            index=id,
        )
        losses["loss"] += losses[cl_loss_AB] + losses[cl_loss_AC] #+losses[cl_loss_BC]
        return losses

#MARK 
    def forward(
        self,
        batch,
    ) -> dict:

        wav = batch["wav"]
        wav_len = batch["wav_len"]
        image = batch["image"]
        text = torch.squeeze(batch['text'])
        id = batch["id"]

        # update device information to clip model
        self.clip.update_device(self.device)

        image_feat = None
        text_feat = None
        audio_feat = None


        modality_list = self.config.model_settings.modality

        if 'audio' in modality_list:
            audio_feat, audio_len = self.forward_audio(wav, wav_len)
            #TODO if no additional transformer 
            if self.parallel_branch is not None:
                parallel_audio_feat = self.parallel_branch(
                    audio_feat=audio_feat,
                    audio_len=audio_len,
                )
                if self.p_branch_proj_net is not None:
                    parallel_audio_feat = self.p_branch_proj_net(parallel_audio_feat)
                audio_feat = parallel_audio_feat 
               
            else:
                audio_feat = audio_feat.sum(1) / audio_feat.shape[1]
                audio_feat = self.audio_proj_net(audio_feat)
            
            audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)


        if 'image' in modality_list:
            image_feat = self.forward_image(image)
            # if self.img_enc_proj_net is not None:
        #     image_feat = self.img_enc_proj_net(image_feat)
            image_feat = F.normalize(image_feat.float())
        if 'text' in modality_list:
            text_feat = self.forward_text(text)
            # if self.text_enc_proj_net is not None:
        #     text_feat = self.text_enc_proj_net(text_feat)
            text_feat = F.normalize(text_feat.float())
        

        
        
        #image = self.clip_preprocess(img)
        #text_feat = F.normalize(self.clip_encoder.encode_text(text).float())
        #image_feat = F.normalize(self.clip_encoder.encode_image(image).float())

        cascaded_audio_feat = None
        parallel_audio_feat = None
        vq_results = None
        keywords = None
       
        
        
        #image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
        #text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        

        losses = {
            "id": id,
            "image_feat": image_feat,
            "audio_feat": audio_feat,
            "text_feat": text_feat,
        }
        log_metrics = {}


        if self.config.model_settings.parallel_objective_weight > 0:
            pass

        log_metrics.update(
            {
                "cl_temp": self.criterion.current_temperature,
            }
        )

        return (
            losses,
            log_metrics,
            {
                "cascaded_audio_feat": cascaded_audio_feat,
                "audio_feat": audio_feat,
                "image_feat": image_feat,
                "text_feat": text_feat,
                "id": id,
                "vq_results": vq_results,
                "keywords": keywords,
            },
        )

    def get_attention_weights(
        self, wav: Union[Tuple[torch.Tensor], List[torch.Tensor]]
    ):
        """get_attention_weights

        For attention map visualization
        Args:
            wav (Union[Tuple[torch.Tensor], List[torch.Tensor]]):

        Returns:
            attention weights
        """
        wav_len = [len(x) for x in wav]
        self.clip.update_device(self.device)
        audio_feat, audio_len = self.forward_audio(wav, wav_len)

        return self.cascaded_branch.getAttentionMap(audio_feat, audio_len)
