"""
why select this code?
Due to the 100-line restriction, I selected this code of the main structure of an machine learning model, 
which represents my typical work and somewhat mirrors my actual coding style.  
"""
class brewclip(Base):
    def __init__(self, cfg):
        super().__init__(cfg)
        modality_list = self.cfg.model_settings.modality 
        self.e2e = self.cfg.model_settings.e2e
        self.pipeline = self.cfg.model_settings.pipeline
        self.use_GT_text = self.cfg.model_settings.use_GT_text
        self.text_proj_net = None   
        self.img_proj_net = None
        self.audio_proj_net = None
        if 'image' or 'text' in modality_list:
            self.clip = CustomCLIP(
            self.cfg.clip,
            )
        if self.e2e:
            self.audio_encoder_type = self.cfg.audio_encoder.type
            self.whisper_encoder = whisperx.load_model(whisper_arch = self.cfg.ASR.name, device = 'cuda', compute_type="float16")
            self.whisper_proj_net = nn.Linear(512, 768)

            if self.cfg.model_settings.audio_branch.projection:
                self.audio_proj_net = Proj_net(cfg,768,768,self.cfg.model_settings.audio_branch.projection_type)

    def forward(
        self,
        batch,
    ) -> dict:
        """
        forward function

        Args:
            batch (dict): batched multimodal data

        Returns:
            dict: {
                "id": id,
                "image_feat": image_feat,
                "audio_feat": final_audio_feat, # actually transcription feat 
                "text_feat": text_feat,
                }
        """
        wav = batch["wav"]
        image = batch["image"]
        if self.use_GT_text:
            transcription = torch.squeeze(batch['text'])
        else:
            transcription =  batch['transcription']
        id = batch["id"]
        # update device information to clip model
        self.clip.update_device(self.device)
        image_feat = None
        text_feat = None
        audio_feat = None
        audio_prompt = None

        # select model type
        if self.cfg.clip.prompt:
            if self.pipeline:
                pip_feat, shallow_prompt, deep_prompt, transcription_output,prompt_embeddings, prompt_audio  = self.clip.encode_text(transcription)
            else:
                shallow_prompt, prompt_audio, deep_prompt = self.clip.get_prompt()
            image_feat = self.clip.encode_image(image,shallow_prompt,deep_prompt)
        else:
            if self.pipeline:
                pip_feat, pip_last_hidden = self.clip.encode_text(transcription)
            image_feat = self.clip.encode_image(image)
        if self.e2e:
            audio_output = self.whisper_encoder.transcribe(wav)
            audio_output = audio_output.to(self.device)
            audio_output = self.whisper_proj_net(audio_output)
            if self.cfg.model_settings.audio_branch.projection_type == "TransformerEncoder":
                prompt_audio = prompt_audio.expand(audio_output.shape[0], -1, -1)
                prompt_feat = torch.cat([prompt_audio, audio_output],dim=1)
                e2e_feat = self.audio_proj_net(prompt_feat,0)
            else:    
                e2e_feat = audio_output[:, 0, :]

        if self.pipeline and self.e2e:
            final_audio_feat = pip_feat + e2e_feat
        elif self.pipeline:
            final_audio_feat = pip_feat
        elif self.e2e:
            final_audio_feat = e2e_feat
        image_feat = F.normalize(image_feat.float())
        final_audio_feat = F.normalize(final_audio_feat.float())
        out = {
            "id": id,
            "image_feat": image_feat,
            "audio_feat": final_audio_feat, # actually transcription feat 
            "text_feat": text_feat,
        }
        return out 
