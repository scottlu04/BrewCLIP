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
        name: str,
        device: str = "cpu",
    ):
        super().__init__()
        assert name in _whisper_models
        self.name = name
        self.device = device
        #self.feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-medium")
        # self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base", task="transcribe")
        # self.processor = WhisperProcessor.from_pretrained("openai/whisper-base", task="transcribe")
        # self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        
        # self.model.config.forced_decoder_ids = None
        # self.model.config.suppress_tokens = []

        self.whisper_asr = pipeline(
         task="automatic-speech-recognition",
            #model="openai/whisper-base",
            model = "facebook/wav2vec2-base-960h",
        #framework="pt",
        batch_size=4,
        #device=0,
        chunk_length_s=30,
        generate_kwargs={"max_new_tokens": 1024},
        )
            #self.freeze_models()


    def freeze_models(self):
        """Freeze Models if required"""
        self.ASR.eval()

    def trainable_params(self) -> list:
        raise NotImplementedError

    def update_device(self, device):
        # since it is a pure nn.Module, it won't update itself
        self.device = device
    
    class DataCollatorSpeechSeq2SeqWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            # convert to tensors
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

            label_features = [{"input_ids": feature["labels"]} for feature in features]
            # pad label ids to the max length in the batch
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

            # if bos token is appended in previous tokenization step,
            # cut bos token here as it's append later anyways
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]

            batch["labels"] = labels

            return batch
    def forward(self, ds) -> torch.Tensor:
        print(ds[0]["audio"])
        wav1,_ = librosa.load(ds[0]["audio"],sr=16_000)
        wav2,_ = librosa.load(ds[1]["audio"],sr=16_000)
        
        t = self.whisper_asr([wav1, wav2, wav2, wav2])
        print(t)

        # #inputs = self.processor(ds[1]["audio"]["array"], return_tensors="pt")
        
        
        # input_features = inputs.input_features

        # generated_ids = self.model.generate(inputs=input_features)

        # transcription = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True,clean_up_tokenization_spaces = True,max_length=448)#[0]
        # print(transcription)
        #print(clean_up_tokenization(transcription))
        # ds = ds.map(self.prepare_dataset, num_proc=2)
        # print(ds[1])
        # #wav, _ = librosa.load(wav)
        # #input_features = self.processor(wav, return_tensors="pt").input_features
        # #input_features = self.feature_extractor(wav, sampling_rate=16_000).input_features[0]
        
        # print(input_features)

        # batch = torch.tensor(input_features).view(1,80,3000)
        # batch["labels"] = self.tokenizer('ssjfs kfjk xxx').input_ids#.view()
        # print(batch["labels"].size())
        # #batch = batch.view(1,80,3000)
        # pred_ids = self.model(batch)
        # pred = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    def prepare_dataset(self, batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = self.tokenizer('sentence').input_ids
        return batch

if __name__ == "__main__": 
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
    audio_dataset = Dataset.from_dict({"audio": [path+f1, path+f2]})#.cast_column("audio", Audio(16000))
    model(audio_dataset)

    
    #model(audio_dataset)
    # wav1, _ = librosa.load(path+f1,sr=16_00)
    # print(torch.tensor(wav1).size())
    
    # wav2, _ = librosa.load(path+f2,sr=16_00)
    # print(torch.tensor(wav2).size())
    # new = pad_sequence([torch.tensor(wav1), torch.tensor(wav1)])
    # model.forward(new)

    #/media/exx/HDD/zhenyulu/flicker/flickr30k_train/flickr30k_train_371902_134.ogg