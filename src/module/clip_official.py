import logging

logger = logging.getLogger(__name__)
import os
import string

import clip
import numpy as np
import torch
from clip.simple_tokenizer import SimpleTokenizer
from PIL import Image
from torch import nn

_clip_models = {
    "RN50",
    "RN101",
    "RN50x4",
    "RN50x16",
    "RN50x64",
    "ViT-B/32",
    "ViT-B/16",
    "ViT-L/14",
}


class ClipModel(nn.Module):
    def __init__(
        self,
        name: str,
        device: str = "cpu",
        image_encoder_trainable: bool = False,
        text_encoder_trainable: bool = False,
        reduce_subword_embbedding: str = None,
        **kwargs,
    ):
        """Official CLIP model.

        Args:
            name (str): Name of CLIP model.
            device (str, optional): Device. Defaults to "cpu".
            image_encoder_trainable (bool, optional): Whether to train the image encoder. Defaults to False.
            text_encoder_trainable (bool, optional): Whether to train the text encoder. Defaults to False.
            reduce_subword_embbedding (str, optional): The reduced vocabulary. Defaults to False
        """
        super().__init__()
        assert name in _clip_models
        self.name = name
        self.device = device

        self.model, self.image_preprocess = clip.load(name, device)

        self.image_encoder_trainable = image_encoder_trainable
        self.text_encoder_trainable = text_encoder_trainable

        self.out_dim = self.model.transformer.width

        self.tokenizer = SimpleTokenizer()

        self.freeze_models()

        self.selected_text_emb_ids = None
   

    def freeze_models(self):
        """Freeze Models if required"""

        if not self.image_encoder_trainable:
            # freeze visual
            for p in self.model.visual.parameters():
                p.requires_grad = False

        if not self.text_encoder_trainable:
            for p in self.model.token_embedding.parameters():
                p.requires_grad = False

            self.model.positional_embedding.requires_grad = False

            for p in self.model.transformer.parameters():
                p.requires_grad = False

            for p in self.model.ln_final.parameters():
                p.requires_grad = False

            self.model.text_projection.requires_grad = False
            self.model.logit_scale.requires_grad = False

    def trainable_params(self) -> list:
        params = []
        if self.image_encoder_trainable:
            params += list(self.model.visual.parameters())
        if self.text_encoder_trainable:
            params += list(self.model.token_embedding.parameters())
            params += [self.model.positional_embedding]
            params += list(self.model.transformer.parameters())
            params += list(self.model.ln_final.parameters())
            params += [self.model.text_projection]

        return params

    def update_device(self, device):
        # since it is a pure nn.Module, it won't update itself
        self.device = device

    def prep_image(self, paths: list) -> torch.Tensor:
        """Prepare image tensor

        Args:
            paths (list): Paths to multiple images

        Returns:
            torch.Tensor: Preprocessed image tensor (B, 3, H, W)
        """
        image_list = []
        for p in paths:
            img = Image.open(p)
            image_list.append(self.image_preprocess(img))
        return torch.stack(image_list, dim=0).to(self.device)

    def prep_text(self, sents: list) -> torch.Tensor:
        """Tokenize text

        Args:
            sents (list): Sentences

        Returns:
            torch.Tensor: _description_
        """
        res = clip.tokenize(sents, truncate=True)
        if self.selected_text_emb_ids is not None:
            for sent in res:
                for i in range(len(sent)):
                    sent[i] = self.original2Reduced[sent[i].item()]
        return res



    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images.

        Args:
            image (torch.Tensor): Images. (B, 3, H, W)

        Returns:
            torch.Tensor: Image features. (B, D)
        """
        return self.model.encode_image(image)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode a batch of sentences.
        Args:
            text (torch.Tensor): Sentences. (B, L)
        Returns:
            torch.Tensor: Text features. (B, D)
        """
        return self.model.encode_text(text)
    
    def deTokenize(self, sents):
        if isinstance(sents, torch.Tensor):
            # print(sents.shape)
            sents = sents.view(*sents.shape[:2]).tolist()
        res = []
        for sent in sents:
            if self.selected_text_emb_ids is not None:
                for i in range(len(sent)):
                    sent[i] = self.reducedl2Original[sent[i]]
            res.append(
                self.tokenizer.decode(sent)
                .replace("<|startoftext|>", "")
                .replace("<|endoftext|>", "")
                .strip()
            )

        return res

    
