import logging
from typing import Any, Dict, List, Union
logger = logging.getLogger(__name__)
import os
import string
import librosa
import whisper
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration,WhisperFeatureExtractor, WhisperTokenizer
from datasets import load_dataset, Audio,Dataset
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

_whisper_models = {
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    "large.en",

}

# # https://huggingface.co/openai/whisper-base
class asr(nn.Module):
    def __init__(
        self,
        model: str,
        name: str,
        device: str = "cpu",
    ):
        super().__init__()
        if  model != 'Whisper':
            raise NotImplementedError
        assert name in _whisper_models
        self.name = name
        self.device = device
        self.model = pipeline(
         task="automatic-speech-recognition",
            model="openai/whisper-" + self.name,
        framework="pt",
        batch_size=32,
        device=0,
        chunk_length_s=30,
        generate_kwargs={"max_new_tokens": 1024},
        )
        print
            #self.freeze_models()


    def freeze_models(self):
        """Freeze Models if required"""
        self.ASR.eval()

    def trainable_params(self) -> list:
        raise NotImplementedError

    def update_device(self, device):
        # since it is a pure nn.Module, it won't update itself
        self.device = device

    def forward2(self, audio):
        # print(ds[0]["audio"])
        # wav1,_ = librosa.load(ds[0]["audio"],sr=16_000)
        # wav2,_ = librosa.load(ds[1]["audio"],sr=16_000)

        audio = list(audio.detach().cpu().numpy())#.cuda()
        transcription = self.model(audio)
        transcription = [ sub['text'] for sub in transcription ]
        return transcription

    def forward(self, audio):
        # print(ds[0]["audio"])
        # wav1,_ = librosa.load(ds[0]["audio"],sr=16_000)
        # wav2,_ = librosa.load(ds[1]["audio"],sr=16_000)

        #audio = list(audio.detach().cpu().numpy())#.cuda()
        #audio = list(audio.detach().cpu().numpy())#.cuda()
        #print(audio)
        audio_ds = Dataset.from_dict({"audio": audio}).cast_column("audio", Audio(16000))
        #print(audio_ds["audio"])
        #transcription = self.model(audio)
        #print(transcription)
        transcriptions = []
        for transcription in self.model(KeyDataset(audio_ds, "audio"),batch_size=32):
            transcriptions.append(transcription['text'])
        #print(transcriptions)
        #transcription = [ sub['text'] for sub in transcription ]
        return transcriptions
if __name__ == "__main__": 
#     tensor([6655, 4242, 3810, 5685, 7378, 6764, 5781, 3693, 1705, 2309, 6420, 2637,
#         6980,  860, 7921,  255, 5833,  396, 1552, 1895, 1087, 2139, 3878, 7343,
#         7335, 3174, 3166, 5351, 3490, 1088, 6039, 1473], device='cuda:0')
# Epoch 0:  34%|█████████████████████████████████████▏                                                                      | 431/1250 [21:17<40:27,  2.96s/it, loss=3.25, v_num=3, loss_step=3.350, c_cl_loss_ai_step=3.350]tensor([3758,  656, 3006, 5511, 6924, 3308, 5129, 3106, 5374, 2255, 4221, 6401,
#          263, 6643, 1779, 6506, 6853, 6227, 5964, 5602, 1693, 2128, 7359, 3092,
#         7129, 6781, 7827, 6735, 3281, 5509, 7884, 1369], device='cuda:0')
    #audio_dataset = Dataset.from_dict({"audio": ["path/to/audio_1", "path/to/audio_2", ..., "path/to/audio_n"]}).cast_column("audio", Audio())
    model = asr("base.en",'cpu')
    from tqdm import tqdm
    # dirs = os.listdir(path)
    #pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
    #dataset = datasets.load_dataset("superb", name="asr", split="test")
    # for dir in tqdm(dirs, desc = 'dirs'):
    # print(dir)
    path = '/media/exx/HDD/zhenyulu/flicker/flickr30k_train/'
    dirs = os.listdir('/media/exx/HDD/zhenyulu/flicker/flickr30k_train/')
    # for file in tqdm(dirs, desc = 'dirs'):
    #     #print(file)
    #     audio_dataset = Dataset.from_dict({"audio": [path+file]}).cast_column("audio", Audio(16000))
    #     #dataset = load_dataset("audiofolder", data_dir=path)
    #     print(audio_dataset[0]["audio"]["array"])
    #     #wav,_ = librosa.load(path+file,sr=16_000)
    #     #print(wav)
    #     model.forward(audio_dataset[0]["audio"]["array"])
    #     break

    f1 =  'flickr30k_train_36979_127.ogg'
    f2 = 'flickr30k_train_371902_134.ogg'
    audio_dataset = Dataset.from_dict({"audio": [path+f1, path+f2]}).cast_column("audio", Audio(16000))
    model(audio_dataset)
    # pipe(KeyDataset(dataset, "file")):
    # print(out)
    

    # tr_set = FlickrDataset(
    #                 split="train",
    #                 # load_image=False,
    #                 # tokenizeText=False,
    #                 # modalities=["audio", "image", "text"],
    #                 **self.cfg.data.dataset,
    #             )
    # train_loader = DataLoader(
    #     tr_set,
    #     batch_size=self.cfg.data.batch_size,
    #     shuffle=True,
    #     num_workers=self.cfg.njobs,
    #     pin_memory=True,
    #     drop_last=True,
    #     collate_fn=collate_general,
    # )
    #             dv_loader = DataLoader(
    #                 dv_set,
    #                 batch_size=self.cfg.data.dev_batch_size,
    #                 shuffle=False,
    #                 num_workers=self.cfg.njobs,
    #                 pin_memory=True,
    #                 drop_last=False,
    #                 collate_fn=collate_general,
    #             )

    #model(audio_dataset)