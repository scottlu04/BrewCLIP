import os
import pandas as pd
from typing import List, Union
import numpy as np
import librosa
import clip
import warnings
import random
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
#1-18 train 19-24 test
class RAVDESS_Dataset(Dataset):
    def __init__(
        self,
        name,
        dataset_root: str = "",
        split: str ='train',
        target_sr: int = 16_000,
        audio_transform=None,
        load_audio: bool = True,
        normalize_waveform: bool = False,
        tokenizeText: bool = True,
        ):
        # Paths for data.
        if split == 'train':
            actor_list = np.arange(1,19)
        elif split == 'test':
            actor_list = np.arange(19,25)
        self.split = split

        self.dataset_root = dataset_root
        self.target_sr = target_sr
        self.load_audio = load_audio
        self.audio_transform = audio_transform
        self.normalize_waveform = normalize_waveform
        self.tokenizeText = tokenizeText
        dir_list = os.listdir(dataset_root)
        dir_list.sort()
        emotion = []
        gender = []
        path = []
        self.data = []
        {
            1:'neutral', 
            2:'neutral', 
            3:'happy', 
            4:'sad', 
            5:'angry', 
            6:'fear', 
            7:'disgust', 
            8:'surprise'
        }
        lable_map = {
            1:0, 
            2:0, 
            3:1, 
            4:2, 
            5:3, 
        }
        for i in dir_list:
            if int(i[-2:]) in actor_list:
                fname = os.listdir(dataset_root + i)
                for f in fname:
                    part = f.split('.')[0].split('-')
                    if int(part[2]) > 5:
                        continue 
                    if int(part[4]) == 1:
                        statement = "Kids are talking by the door"
                    else:
                        statement = "Dogs are sitting by the door"
                    
                    # temp = int(part[6])
                    # if temp%2 == 0:
                    #     temp = "female"
                    # else:
                    #     temp = "male"
                    # gender.append(temp)
                    # path.append(dataset_root + i + '/' + f)
                    _entry = {
                        "wav": dataset_root + i + '/' + f,
                        "id":lable_map[int(part[2])],
                        "transcription":statement
                        }
                    self.data.append(_entry)
          #      print(path)
    def noise(self, data):
        noise_amp = 0.035*np.random.uniform()*np.amax(data)
        data = data + noise_amp*np.random.normal(size=data.shape[0])
        return data

    def stretch(self, data, rate=0.8):
        return librosa.effects.time_stretch(data, rate)

    def shift(self, data):
        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        return np.roll(data, shift_range)

    def pitch(self, data, sampling_rate, pitch_factor=0.7):
        return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)
    def _LoadAudio(self, path: str):
        """Load audio from file

        Args:
            path (str): Path to waveform.

        Returns:
            torch.FloatTensor: Audio features.
        """
        
        if self.load_audio:
            waveform, sample_rate = librosa.load(path, sr=self.target_sr)
            if self.split == 'train':
                ind = random.randint(0, 3)
                #print(ind)
                if ind == 0:
                    waveform = self.noise(waveform)
                elif ind == 1:
                    waveform = self.stretch(waveform)
                elif ind == 2:
                    waveform = self.pitch(waveform, sample_rate)
            if self.audio_transform is not None:
                audio = self.audio_transform(waveform)
            else:
                audio = torch.FloatTensor(waveform)
            if self.normalize_waveform:
                audio = F.layer_norm(audio, audio.shape)
        else:
            audio = path

        return audio
    def _TokenizeText(self, texts: Union[str, List[str]]):
        if self.tokenizeText:
            return clip.tokenize(texts=texts, truncate=True)
            #return clip.tokenize(texts=texts, context_length=77, truncate=True)
        else:
            return texts
    def __getitem__(self, index):
        ret_dict = {}
        if "wav" in self.data[index]:
            audio_feat = self._LoadAudio(self.data[index]["wav"])
            ret_dict["wav"] = audio_feat
            ret_dict["wav_path"] = self.data[index]["wav"]
        if "id" in self.data[index]:
            ret_dict["id"] = self.data[index]["id"]
        if "transcription" in self.data[index]:
            transcription = self._TokenizeText(self.data[index]["transcription"])
            ret_dict["transcription"] = transcription
            #ret_dict["transcription"] = self.data[index]["transcription"]
        assert len(ret_dict) > 0, "dataset getitem must not be empty"

        return ret_dict
    def __len__(self):
        return len(self.data)
    

# RAV = "/media/exx/HDD/zhenyulu/ravdess/"
# data = RAVDESS(RAV)
# data[1]
