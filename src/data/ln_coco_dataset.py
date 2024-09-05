from PIL import Image
from torch.utils.data import Dataset
import torch
import os

from .base_dataset import BaseDataset
from .localized_narratives import DataLoader as LocNarrDataLoader
import clip
import logging

logger = logging.getLogger(__name__)

class CoCoDataset():
    
    def __init__(self, data_dir, split):
        self.data_dir = data_dir
        self.split = split
        
    def __getitem__(self, index):
        fname = format(index, '012d') # could probably replace this with some sort of regex thing for robustness
        fpath = self.data_dir + '/' + self.split + '2017/' + fname + '.jpg'
        return fpath
    
class HelpfulWrapper():
    def __init__(self, locnarr_dataset):
        self.locnarr_dataset = locnarr_dataset
    
    # it's good for us to wrap it like this instead of just pre-caching a to_return list which can be indexed normally
    # because it lets us greedily download audio files
    def __getitem__(self, idx):
        annot = self.locnarr_dataset.locnarr_annot_loader[idx]
        #print(annot)
        to_return = dict()
        to_return['text'] = annot.caption
        to_return['wav'] = self.locnarr_dataset.locnarr_annot_loader.get_audioclip(annot.voice_recording)
        to_return['image'] = self.locnarr_dataset.coco_dataset[int(annot.image_id)]
        to_return['id'] = int(annot.image_id)
        return to_return
    
    def __len__(self):
        return len(self.locnarr_dataset.locnarr_annot_loader)
    
class LN_COCO_Dataset(BaseDataset):
    def __init__(self, clip_image_transform=None, **kwargs):
        super().__init__(**kwargs)
        if clip_image_transform is not None:
            logger.info(
                "Load clip ({}) for image transform".format(clip_image_transform)
            )
            _, image_transform = clip.load(clip_image_transform, "cpu")
            self.image_transform = image_transform
        data_dir = kwargs['dataset_root']
        split = kwargs['split']
        if split == 'test' or split == 'dev':
            split = "val"
        self.coco_dataset = CoCoDataset(data_dir + '/mscoco_img', split) # data/coco/coco_localnarr_audio
        self.locnarr_annot_loader = LocNarrDataLoader(data_dir + '/coco_localnarr_audio', f'coco_{split}')
        self.data = HelpfulWrapper(self) # because BaseDataset expects to index this value to get what it wants, bleh
        #self.annots = self.locnarr_annot_loader.load_annotations(f'coco_{split}')
    def add_transcription(self, aud_transcription_file, num_sample_load):
        aud2transcription = {}
        missing_count  = 0
        with open(os.path.join(self.dataset_root, aud_transcription_file), "r") as fp:
            line_count = 0
            for line in fp:
                line_count +=1
                #print(line_count)
                audio_name, transcription = line.split('#')
                # try:
                #     audio_name, transcription = line.split('# ')
                # except:
                #     audio_name, transcription = line.split('#. ')
                #print(audio_name)
                #print(transcription)
                aud2transcription[audio_name] = transcription
                if line_count == num_sample_load:
                    break
        data =  []
        for ind, x in enumerate(self.data):
            #print(x['wav'])
            ##path = os.path.join(self.dataset_root,'flickr_audio',"wavs", x['wav'] +'.wav')
            name  = x['wav'].split("/")[-1][:-4]
            #print(name)
            if name in aud2transcription:
                #x.update( {"transcription":aud2transcription[name]})
                _entry = {
                    'wav': x['wav'],
                    'transcription': aud2transcription[name],
                    'image': x['image'],
                    'id': x['id'],
                    'text': x['text']
                    }
                data.append(_entry)
        self.data = data
        assert len(self.data) == num_sample_load
        print(f'loaded {num_sample_load} samples')
        #print(f'Missing {missing_count} transcriptions.')

    def add_transcription_full(self, aud_transcription_file):
        aud2transcription = {}
        missing_count  = 0
        with open(os.path.join(self.dataset_root, aud_transcription_file), "r") as fp:
            line_count = 0
            for line in fp:
                line_count +=1
                #print(line_count)
                audio_name, transcription = line.split('#')
                # try:
                #     audio_name, transcription = line.split('# ')
                # except:
                #     audio_name, transcription = line.split('#. ')
                #print(audio_name)
                #print(transcription)
                aud2transcription[audio_name] = transcription
        data =  []
        for ind, x in enumerate(self.data):
            #print(x['wav'])
            ##path = os.path.join(self.dataset_root,'flickr_audio',"wavs", x['wav'] +'.wav')
            name  = x['wav'].split("/")[-1][:-4]
            #print(path)
            #print(name)
            if name in aud2transcription:
                #x.update( {"transcription":aud2transcription[name]})
                _entry = {
                    'wav': x['wav'],
                    'transcription': aud2transcription[name],
                    'image': x['image'],
                    'id': x['id'],
                    'text': x['text']
                    }
                data.append(_entry)
                #x['transcription'] = aud2transcription[name]
                #continue
               # print("True")
            else:
                #print("remove this instance", x['wav'])
                #del self.data[ind]
                #print("Missing")
                missing_count+=1
        self.data = data
        print(f'Missing {missing_count} transcriptions.')