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
from transformers import AutoTokenizer,AutoProcessor, CLIPTextModel,CLIPVisionModel, CLIPModel,CLIPTextModelWithProjection, CLIPVisionModelWithProjection

import sys
sys.path.append('..');
# from ..base import OrderedNamespace
from module import (
    ClipModel,
    FairseqSpeechEncoder_Hubert,
    MLPLayers,
    losses,
    mutualRetrieval,
    TransformerEncoder
)
from optim import get_scheduler
from module.kw_modules import TransformerModels
# from ..module.speechclip_c_modules import vector_quantizers
# from ..module.speechclip_c_modules.kw_bn import Kw_BatchNorm
# from ..optim import get_scheduler
from util import get_keypadding_mask
from .base_model import BaseLightningModel
from .base import Base
__all__ = ["e2e"]
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
class cross_attension_net(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
        # self.mod_encoder1 = TransformerEncoder(embed_dim=embed_dim, 96
        #                             num_heads=self.num_heads,8
        #                             layers=max(self.layers, layers), 1
        #                             attn_dropout=attn_dropout, 0.1
        #                             relu_dropout=self.relu_dropout, 0.1
        #                             res_dropout=self.res_dropout,0.1
        #                             embed_dropout=self.embed_dropout, 0.25
        #                             attn_mask=self.attn_mask) True
        #TODO abstract
        # self.aud_cls = self._create_cls()
        # self.img_cls = self._create_cls()
        self.linear  = nn.Linear(768, 512)
        self.aud_encoder = TransformerEncoder(embed_dim=512)
        self.img_encoder = TransformerEncoder(embed_dim=512)
    
    def _create_cls(self):
        # first cls for parallel objective
        return torch.nn.Parameter(
            torch.randn(
                [
                    1,
                    1,
                    self.cfg.model_settings.transformer_args.d_model,
                ]
            )
        )       
    def forward(self, audio_feat, img_feat, text_feat =None):
        
        
            #audio_len = audio_feat.shape[1]
        img_feat = self.linear(img_feat)
        audio_feat = audio_feat.permute(1, 0, 2)
        img_feat = img_feat.permute(1, 0, 2)
        #print('>>>>>>>>>>>>>>>>>>')
        #print(img_feat.size())
        # feat_aud = self.aud_encoder(audio_feat, img_feat, img_feat)  
        # feat_img = self.img_encoder(img_feat, audio_feat, audio_feat)
        feat_aud = self.aud_encoder(audio_feat, audio_feat, audio_feat)  
        feat_img = self.img_encoder(img_feat, img_feat, img_feat)
        feat_aud = feat_aud.sum(0) / feat_aud.shape[0]
        feat_img = feat_img.sum(0) / feat_img.shape[0]
        #print(feat_aud.size())
        return  feat_aud, feat_img

    # def forward2(self, audio_feat, img_feat, text_feat =None):
        
    #     #audio
    #     bsz, total_max_len = (
    #             audio_feat.size(0),
    #             audio_feat.size(1) + 1,
    #         )
    #     aud_cls = torch.cat([self.aud_cls] * bsz, dim=0)
    #     audio_feat = torch.cat([aud_cls, audio_feat], dim=1)

    #     #text
    #     bsz, total_max_len = (
    #             img_feat.size(0),
    #             img_feat.size(1) + 1,
    #         )
    #     img_cls = torch.cat([self.img_cls] * bsz, dim=0)
    #     img_feat = torch.cat([aud_cls, img_feat], dim=1)
    #         #audio_len = audio_feat.shape[1]

    #     out = self.net(src=src, key_padding_mask=key_padding_mask)
    #     #out = self.net(src=src)
    #     out = out[:, :1].reshape(-1, self.in_dim)
    #     out = self.linear_proj(out)
    #     # if hasattr(self, "linear_proj"):
    #     #     out = self.linear_proj(out)
    #     feat_aud = self.mod_encoder1(feat_aud, feat_img, feat_img)  

    #     feat_img = self.mod_encoder2(feat_img, feat_aud, feat_aud)
    #     return  feat_aud, feat_img


class Proj_net(nn.Module):
    """
    proj-net
    """

    def __init__(self, cfg, in_dim: int, out_dim: int, projection_type: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.projection_type = projection_type
 

        # select the transformer structure for main architecture for parallel branch
        
        if projection_type == 'Linear':
            self.net = nn.Linear(self.in_dim, self.out_dim)
        else:
            assert hasattr(
                TransformerModels, self.cfg.model_settings.transformer_type
            )
            self.net = getattr(
                TransformerModels, self.cfg.model_settings.transformer_type
            )(**self.cfg.model_settings.transformer_args)

            self.cls = self._create_cls()
            self.linear_proj = nn.Linear(self.in_dim, self.out_dim)
            # if self.need_projection:
            #     self.linear_proj = nn.Linear(self.audio_dim, self.out_dim)
        

    def _create_cls(self):
        # first cls for parallel objective
        return torch.nn.Parameter(
            torch.randn(
                [
                    1,
                    1,
                    self.cfg.model_settings.transformer_args.d_model,
                ]
            )
        )

    def forward(
        self, 
        audio_feat: torch.Tensor,
        audio_len
    ) -> torch.Tensor:
       
        if self.projection_type == "Linear":
            audio_feat = audio_feat.sum(1) / audio_feat.shape[1]
            out = self.net(audio_feat)
        #MHA model
        else:
            bsz, total_max_len = (
                audio_feat.size(0),
                audio_feat.size(1) + 1,
            )
            cls = torch.cat([self.cls] * bsz, dim=0)
            src = torch.cat([cls, audio_feat], dim=1)
            #audio_len = audio_feat.shape[1]
            #TODO key_padding_mask
            key_padding_mask = get_keypadding_mask(
                max_length=total_max_len,
                data_lens=audio_len + 1,
            )

            out = self.net(src=src, key_padding_mask=key_padding_mask)
            #out = self.net(src=src)
            out = out[:, :1].reshape(-1, self.in_dim)
            out = self.linear_proj(out)
            # if hasattr(self, "linear_proj"):
            #     out = self.linear_proj(out)

        return out
    def forward_hidden(
        self, 
        audio_feat: torch.Tensor,
        audio_len
    ) -> torch.Tensor:
       
        if self.projection_type == "Linear":
            audio_feat = audio_feat.sum(1) / audio_feat.shape[1]
            out = self.net(audio_feat)
        #MHA model
        else:
            bsz, total_max_len = (
                audio_feat.size(0),
                audio_feat.size(1) + 1,
            )
            cls = torch.cat([self.cls] * bsz, dim=0)
            src = torch.cat([cls, audio_feat], dim=1)
            #audio_len = audio_feat.shape[1]
            #TODO key_padding_mask
            key_padding_mask = get_keypadding_mask(
                max_length=total_max_len,
                data_lens=audio_len + 1,
            )

            out = self.net(src=src, key_padding_mask=key_padding_mask)
            #out = self.net(src=src)
            out = out[:, :1].reshape(-1, self.in_dim)
            out = self.linear_proj(out)
            # if hasattr(self, "linear_proj"):
            #     out = self.linear_proj(out)

        return out

class e2e(Base):

    def __init__(self, cfg):
        super().__init__(cfg)
        #self.save_hyperparameters()
        # select audio_encoder type
        modality_list = self.cfg.model_settings.modality 
        self.text_proj_net = None   
        self.img_proj_net = None
        self.cross_attension_net =None
        self.final_aud_proj = None
        if 'image' or 'text' in modality_list:
            # self.clip = ClipModel(
            # **self.cfg.clip,
            # )
            
            self.text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
            self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.text_model.eval()

            self.img_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.img_model.eval()

            if 'text' in modality_list and self.cfg.model_settings.image_branch.projection:
                self.text_proj_net = Proj_net(cfg,512,512,self.cfg.model_settings.text_branch.projection_type)            
            if 'image' in modality_list and  self.cfg.model_settings.image_branch.projection:
                self.img_proj_net = Proj_net(cfg,512,512,self.cfg.model_settings.image_branch.projection_type)
                # if self.cfg.model_settings.text_branch.projection_type == 'Linear':
                #     self.img_proj_net = nn.Linear(512, 512)
                # elif self.cfg.model_settings.audio_branch.projection_type == 'transformer':
                #     #TODO: transformer or multihead att
                #     raise NotImplementedError     
                
        if 'audio' in modality_list:
            self.audio_encoder_type = self.cfg.audio_encoder.type
            if self.audio_encoder_type == "FairseqHubert":
                self.audio_encoder = FairseqSpeechEncoder_Hubert(**self.cfg.audio_encoder)
            self.audio_proj_net = Proj_net(cfg,self.audio_encoder.out_dim,512,self.cfg.model_settings.audio_branch.projection_type)

            #self.transcription_proj_net = Proj_net(cfg,512,512,"Linear")
            # if self.cfg.model_settings.audio_branch.projection_type == 'Linear':
            #     self.audio_proj_net = nn.Linear(self.audio_embd_dim, 512)
            # elif self.cfg.model_settings.audio_branch.projection_type == 'transformer':
            #     #TODO: transformer or multihead att
            #     raise NotImplementedError
            # else:
            #     raise NotImplementedError
        self.cross_attension_net = cross_attension_net(cfg)
        self.final_aud_proj = nn.Linear(512, 512)
    def forward(
        self,
        batch,
    ) -> dict:
        #TODO Change back to librosa.laod, to support hubert extraction
        #print(batch)
        wav = batch["wav"]
       # wav_len = batch["wav_len"]
        image = batch["image"]
        text = torch.squeeze(batch['text'])
        id = batch["id"]
        #print(batch)
        
        # update device information to clip model
        #self.clip.update_device(self.device)

        image_feat = None
        text_feat = None
        audio_feat = None


        modality_list = self.cfg.model_settings.modality

        if 'audio' in modality_list:
            if self.cfg.mode == 'asr':
                transcription =  batch['transcription']
                #print(transcription.size())
                #transcription_feat = self.forward_text(transcription)
                #transcription_feat = self.tokenizer(list(transcription), padding=True, return_tensors="pt")
                #transcription_feat = self.text_model(**transcription_feat)
                transcription_feat3 = self.text_model(transcription.squeeze())
                transcription_feat = transcription_feat3.text_embeds
                transcription_feat2 = transcription_feat3.last_hidden_state 
                transcription_feat = F.normalize(transcription_feat.float())
                audio_feat = transcription_feat
                if self.cfg.residual:
                    wav_len = batch["wav_len"]
                    resi_audio_feat, audio_len = self.forward_audio(wav, wav_len)
                    resi_audio_feat = self.audio_proj_net(resi_audio_feat,audio_len)
                    resi_audio_feat = F.normalize(resi_audio_feat.float())
               # audio_feat += resi_audio_feat
                #print(transcription_feat2.size())
                #transcription_feat = transcription_feat / transcription_feat.norm(dim=-1, keepdim=True)
            elif self.cfg.mode == 'e2e':
                wav_len = batch["wav_len"]
                audio_feat, audio_len = self.forward_audio(wav, wav_len)
                audio_feat = self.audio_proj_net(audio_feat,audio_len)
            #TODO if no additional transformer 
            # if self.cfg.model_settings.image_branch.projection_type == 'Linear':
            #     audio_feat = audio_feat.sum(1) / audio_feat.shape[1]
            #     audio_feat = self.audio_proj_net(audio_feat)
            # else:
            #     raise NotImplementedError
            #audio_feat = audio_feat / audio_feat.norm(dim=-1, keepdim=True)


        if 'image' in modality_list:
            #image_feat = self.processor(images=image, return_tensors="pt")
            #image_feat =  self.img_model(**image_feat)
            image_feat3 =  self.img_model(image)
            image_feat = image_feat3.image_embeds 
            image_feat2 = image_feat3.last_hidden_state  
           # print(image_feat2.size())
            #image_feat = self.forward_image(image)
            if self.img_proj_net is not None:
              image_feat = self.img_proj_net(image_feat)
            image_feat = F.normalize(image_feat.float())
            #print(image_feat.size())
        if 'text' in modality_list:
            text_feat = self.text_model(text.squeeze())
            text_feat = text_feat.text_embeds
            #    transcription_feat2 = transcription_feat3.last_hidden_state 
           #     audio_feat = transcription_feat
         #   text_feat = self.forward_text(text)
            if self.text_proj_net is not None:
                text_feat = self.text_proj_net(text_feat)
            text_feat = F.normalize(text_feat.float())
            
        audio_feat2, image_feat2 =  self.cross_attension_net(transcription_feat2, image_feat2)
        image_feat2 = F.normalize(image_feat2.float())
        audio_feat2 = F.normalize(audio_feat2.float())
        audio_feat2 += resi_audio_feat
        audio_feat2 = self.final_aud_proj(audio_feat2)
        audio_feat2 = F.normalize(audio_feat2.float())
        # print(image_feat2.size())
        # print(audio_feat2.size())
        out = {
            "id": id,
            "image_feat": image_feat2,
            "audio_feat": audio_feat2, # actually transcription feat 
            "text_feat": text_feat,
        }
        # log_metrics = {}

        # log_metrics.update(
        #     {
        #         "cl_temp": self.criterion.current_temperature,
        #     }
        # )
        return out
    

 
    def validation_epoch_end(self, outputs: list):
        """validation_epoch_end

        Args:
            outputs (list): list of aggregated results
        """
        # if keywords is in the input, calculate keyword related metrics
    
        #MARK
        #outputs = outputs['others']
        #detach output
        for i in range(len(outputs)):
            for k in outputs[i]:
                if isinstance(outputs[i][k], torch.Tensor):
                    outputs[i][k] = outputs[i][k].detach().cpu()
        modality_list = self.cfg.model_settings.modality 
        assert len(modality_list) >= 2
        #TODO for flicker8k only
        all_ids = torch.cat([x["id"] for x in outputs], dim=0)
        feat_dict = []
        for mod in modality_list:
            feat_name = mod+"_feat"
            all_feats = torch.cat([x[feat_name] for x in outputs], dim=0)

            # id_feat_pairs = {_id.item(): _x for _id, _x in zip(all_ids, all_feats)}
            # all_feats = torch.stack([x for _, x in id_feat_pairs.items()], dim=0)
            # all_feats_id = torch.LongTensor(list(id_feat_pairs.keys()))
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
        all_C_feats = None
        if len(modality_list) == 3:
            all_C_feats = feat_dict[2][1]
            all_C_feats_id = feat_dict[2][2]
            feat_C_name = feat_dict[2][0]
        recall_results_AB,recall_results_BA = self.Bi_Retrieval(all_A_feats,all_B_feats,all_A_feats_id,all_B_feats_id,feat_A_name,feat_B_name)
        recall_results_AB = list(recall_results_AB.values())
        recall_results_BA = list(recall_results_BA.values())
        
        outfile = f"& {recall_results_AB[0] :.1f} & {recall_results_AB[1] :.1f} & {recall_results_AB[2] :.1f} "
        outfile += f"& {recall_results_BA[0] :.1f} & {recall_results_BA[1] :.1f} & {recall_results_BA[2] :.1f} "
        if all_C_feats != None:
            recall_results_AC,recall_results_CA = self.Bi_Retrieval(all_A_feats,all_C_feats,all_A_feats_id,all_C_feats_id,feat_A_name,feat_C_name)
            recall_results_BC,recall_results_CB = self.Bi_Retrieval(all_B_feats,all_C_feats,all_B_feats_id,all_C_feats_id,feat_B_name,feat_C_name)
            recall_results_AC = list(recall_results_AC.values())
            recall_results_CA = list(recall_results_CA.values())
            recall_results_BC = list(recall_results_BC.values())
            recall_results_CB = list(recall_results_CB.values())
            outfile += f"& {recall_results_AC[0] :.1f} & {recall_results_AC[1] :.1f} & {recall_results_AC[2] :.1f} "
            outfile += f"& {recall_results_CA[0] :.1f} & {recall_results_CA[1] :.1f} & {recall_results_CA[2] :.1f} "
            outfile += f"& {recall_results_BC[0] :.1f} & {recall_results_BC[1] :.1f} & {recall_results_BC[2] :.1f} "
            outfile += f"& {recall_results_CB[0] :.1f} & {recall_results_CB[1] :.1f} & {recall_results_CB[2] :.1f}"
        
        print(outfile)
        #self.bi_Retrieval(all_B_feats,all_C_feats,all_B_feats_id,all_C_feats_id,feat_B_name,feat_C_name)

    def compute_loss(self, input_feats: dict):
        """compute the loss here

        Args:
            input_feats (dict): the feats required for computing loss
        """
        assert isinstance(input_feats, dict)
 
        modality_list = self.cfg.model_settings.modality
        # feat_list =[]
        # for mod in modality_list:
        #     feat_name = mod+"_feat"
        #     feat = input_feats[feat_name].float()
        #     feat_list.append([feat_name,feat])

        assert len(modality_list) >= 2
        feat_A_name = modality_list[0]+"_feat"
        assert feat_A_name in input_feats 
        feat_A = input_feats[feat_A_name].float()
        
        feat_B_name = modality_list[1]+"_feat"
        assert feat_B_name in input_feats 
        feat_B = input_feats[feat_B_name].float()
        if len(modality_list) > 2:
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
        losses["loss"] += losses[cl_loss_AB]
        if len(modality_list) > 2:
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
            losses["loss"] += losses[cl_loss_AC] + losses[cl_loss_BC]
        self.log_dict(
            losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return losses["loss"]


    def getTrainableParams(self) -> list:
        """getTrainableParams

        return trainable parameter list
        children class should return their additional trainable parameters

        Returns:
            list: list of trainable parameters
        """
        my_params = []
       
        
        if hasattr(self, "audio_encoder"):
            #my_params += self.audio_encoder.trainable_params()
            #my_params += list(self.audio_proj_net.parameters())
            my_params += list(self.criterion.parameters())
        
        if self.cross_attension_net != None:
            my_params += self.cross_attension_net.parameters()
            print("add cross_attension_net")

        if self.final_aud_proj != None:
            my_params += self.final_aud_proj.parameters()
            print("add final_aud_proj")
        if self.text_proj_net != None:
            my_params += self.text_proj_net.parameters()
            print("add text")

        if self.audio_proj_net != None:
            my_params += list(self.audio_proj_net.parameters()) #self.audio_proj_net.parameters()
            print("add audio")
        if self.img_proj_net != None:
            my_params += self.img_proj_net.parameters()

        #my_params += self.clip.trainable_params()

        return my_params
 