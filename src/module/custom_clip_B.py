import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

#from clip import clip
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .clip import clip

_tokenizer = _Tokenizer()


def load_clip(cfg):
    print(cfg)
    backbone_name = cfg.name #"ViT-B/16" #cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url=url)
    design_details = cfg.design_details
    model = torch.jit.load(model_path, map_location="cuda").eval()
    # try:
    #     # loading JIT archive
    #     model = torch.jit.load(model_path, map_location="cuda").eval()
    #     state_dict = None

    # except RuntimeError:
    #     state_dict = torch.load(model_path, map_location="cpu")
    # design_details = {"trainer": 'MaPLe',
    #                   "vision_depth": 0,
    #                   "language_depth": 0, "vision_ctx": 0,
    #                   "language_ctx": 0,
    #                 #   "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    #                  "maple_length": cfg.n_ctx}
    model = clip.build_model(model.state_dict(), design_details)
    model.float()
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        last_hidden_state = x
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x, last_hidden_state

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.n_ctx = cfg.design_details.maple_length#cfg.TRAINER.MAPLE.N_CTX
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        # Default is 1, which is compound shallow prompting
        self.compound_prompts_depth = cfg.design_details.compound_prompts_depth  # max=12, but will create 11 such shared prompts
        # random initialization
        ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
     
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.proj = nn.Linear(self.ctx_dim, 768)
        #self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        single_layer = nn.Linear(self.ctx_dim, 768)
        self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

    def construct_prompts(self, tokens):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(len(tokens), -1, -1)

        tokens = tokens.squeeze()
        # prompt_embedding = torch.empty(len(tokens), 77, self.ctx_dim, dtype=self.dtype)
        prompt_token = torch.zeros(len(tokens), 77).cuda()
       # print(tokens.device)
        #print(self.clip_model.device)
        
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokens).type(self.dtype)
            #print(embedding.size())
        for i, token in enumerate(tokens):
            #eos ind 
            ind = torch.argmax(token, -1)

            prompt_token[i][0] = token[0]
            sent_token = token[1:]
            #print(sent_token)
            trucated_sent_len = 77 - self.n_ctx - 1
            prompt_token[i][self.n_ctx + 1:] = sent_token[:trucated_sent_len]
            if prompt_token[i][-1] != 0:
                prompt_token[i][-1] = token[ind]

            #prompt_embedding[i][0] = embedding[i][0]
            # if ind == 76:
            #     #remove the last few tokens depends on the length of context 
            #     prompt_token[i][self.n_ctx + 1: self.n_ctx + ind-len(self.ctx)] = token[1:ind-len(self.ctx)]
            #     prompt_token[i][ind] = token[ind]
            # else:
            #     prompt_token[i][self.n_ctx + 1: self.n_ctx + ind] = token[1:ind]
            #     prompt_token[i][self.n_ctx + ind] = token[ind]
            prompt_embedding = self.clip_model.token_embedding(prompt_token.to(torch.int64)).type(self.dtype)
            prompt_embedding[:,1:1+self.n_ctx,:] = ctx
            #print(prompt_embedding2.mean(2)[i])
        return prompt_embedding, prompt_token
    def forward(self,text):

        prompt_embeddings, prompt_tokens = self.construct_prompts(text)
        # # Before returning, need to transform
        # # prompts to 768 for the visual side
        visual_deep_prompts = []
        for index, layer in enumerate(self.compound_prompt_projections):
            visual_deep_prompts.append(layer(self.compound_prompts_text[index]))
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        return prompt_embeddings, prompt_tokens,self.ctx, self.proj(self.ctx), self.compound_prompts_text, visual_deep_prompts   # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.clip_model = load_clip(cfg)
        self.prompt = cfg.prompt
        if cfg.prompt:
            self.prompt_learner = MultiModalPromptLearner(cfg, self.clip_model)
            self.text_encoder = TextEncoder(self.clip_model)
        #self.freeze_models()        
        self.image_encoder = self.clip_model.visual
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype
    

    def freeze_models(self):
        """Freeze Models if required"""
        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")



    def encode_text(self, text):
        text = text.squeeze()
        if self.prompt:
            #pass 
            prompt_embeddings, prompt_tokens, ctx_audio, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(text)
            #print(prompt_tokens[0])
            text_features,text_last_hidden = self.text_encoder(prompt_embeddings, prompt_tokens, deep_compound_prompts_text)
            #print(text_features.size())
           
            #print(deep_compound_prompts_vision.size())

            return text_features, shared_ctx, deep_compound_prompts_vision,ctx_audio, text_last_hidden
        else:
            text_features =  self.clip_model.encode_text(text)
            #text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features
    def encode_image(self, image, shallow_prompt = None, deep_prompt = None):
        if self.prompt:
            #pass 
            image_features = self.image_encoder(image.type(self.dtype), shallow_prompt, deep_prompt)
        else:
            assert shallow_prompt == None, deep_prompt == None
            image_features = self.image_encoder(image.type(self.dtype))
        
        return image_features
    def encode(self, image, text):
        #No prompt token is used in this mode
        assert self.prompt == False
        #print(text.size())
        text = text.squeeze()
        text_features =  self.clip_model.encode_text(text)
        image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return image_features,text_features
    
    def encode_prompt(self, image, text):
        # tokenized_prompts = self.tokenized_prompts
        # logit_scale = self.logit_scale.exp()

        prompt_embeddings, prompt_tokens, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner(text)
        #print()
        text_features = self.text_encoder(prompt_embeddings, prompt_tokens, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return image_features,text_features
    
    def update_device(self, device):
        # since it is a pure nn.Module, it won't update itself
        self.device = device

    def trainable_params(self) -> list:
        params = []
        if self.prompt:
            for name, param in self.prompt_learner.named_parameters():
                    #print(param)
                    if 'clip_model' not in name:
                        print(name)
                        param.requires_grad = True
                    params += [param]
            #params += self.prompt_learner.parameters()
            # name_to_update = "prompt_learner"
            # for name, param in self.clip.named_parameters():
            #     if name_to_update not in name:
            #         # Make sure that VPT prompts are updated
            #         params += list(param)
            # Double check
            # enabled = set()
            # for name, param in self.clip.named_parameters():
            #     if param.requires_grad:
            #         enabled.add(name)
        # if self.image_encoder_trainable:
        #     params += list(self.model.visual.parameters())
        # if self.text_encoder_trainable:
        #     params += list(self.model.token_embedding.parameters())
        #     params += [self.model.positional_embedding]
        #     params += list(self.model.transformer.parameters())
        #     params += list(self.model.ln_final.parameters())
        #     params += [self.model.text_projection]
        #params.requires_grad = True
        return params 


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# class MaPLe(nn.Module):

#     def build_model(self):
#         cfg = self.cfg
#         classnames = self.dm.dataset.classnames

#         print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
#         clip_model = load_clip_to_cpu(cfg)

#         if cfg.TRAINER.MAPLE.PREC == "fp32" or cfg.TRAINER.MAPLE.PREC == "amp":
#             # CLIP's default precision is fp16
#             clip_model.float()

#         print("Building custom CLIP")
#         self.model = CustomCLIP(cfg, classnames, clip_model)

#         print("Turning off gradients in both the image and the text encoder")
#         name_to_update = "prompt_learner"

#         for name, param in self.model.named_parameters():
#             if name_to_update not in name:
#                 # Make sure that VPT prompts are updated
#                 if "VPT" in name:
#                     param.requires_grad_(True)
#                 else:
#                     param.requires_grad_(False)

#         # Double check
#         enabled = set()
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 enabled.add(name)
#         print(f"Parameters to be updated: {enabled}")

#         if cfg.MODEL.INIT_WEIGHTS:
#             load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

#         self.model.to(self.device)
#         # NOTE: only give prompt_learner to the optimizer
#         self.optim = build_optimizer(self.model, cfg.OPTIM)
#         self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
#         self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

#         self.scaler = GradScaler() if cfg.TRAINER.MAPLE.PREC == "amp" else None

#         # Note that multi-gpu training could be slow because CLIP's size is
#         # big, which slows down the copy operation in DataParallel
#         device_count = torch.cuda.device_count()
#         if device_count > 1:
#             print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
#             self.model = nn.DataParallel(self.model)

#     def forward_backward(self, batch):
#         image, label = self.parse_batch_train(batch)

#         model = self.model
#         optim = self.optim
#         scaler = self.scaler

#         prec = self.cfg.TRAINER.MAPLE.PREC
#         if prec == "amp":
#             with autocast():
#                 loss = model(image, label)
#             optim.zero_grad()
#             scaler.scale(loss).backward()
#             scaler.step(optim)
#             scaler.update()
#         else:
#             loss = model(image, label)
#             optim.zero_grad()
#             loss.backward()
#             optim.step()

#         loss_summary = {"loss": loss.item()}

#         if (self.batch_idx + 1) == self.num_batches:
#             self.update_lr()

#         return loss_summary

#     def parse_batch_train(self, batch):
#         input = batch["img"]
#         label = batch["label"]
#         input = input.to(self.device)
#         label = label.to(self.device)
#         return input, label

#     def load_model(self, directory, epoch=None):
#         if not directory:
#             print("Note that load_model() is skipped as no pretrained model is given")
#             return

#         names = self.get_model_names()

#         # By default, the best model is loaded
#         model_file = "model-best.pth.tar"

#         if epoch is not None:
#             model_file = "model.pth.tar-" + str(epoch)

#         for name in names:
#             model_path = osp.join(directory, name, model_file)

#             if not osp.exists(model_path):
#                 raise FileNotFoundError('Model not found at "{}"'.format(model_path))

#             checkpoint = load_checkpoint(model_path)
#             state_dict = checkpoint["state_dict"]
#             epoch = checkpoint["epoch"]

#             # Ignore fixed token vectors
#             if "prompt_learner.token_prefix" in state_dict:
#                 del state_dict["prompt_learner.token_prefix"]

#             if "prompt_learner.token_suffix" in state_dict:
#                 del state_dict["prompt_learner.token_suffix"]

#             print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
#             # set strict=False
#             self._models[name].load_state_dict(state_dict, strict=False)
