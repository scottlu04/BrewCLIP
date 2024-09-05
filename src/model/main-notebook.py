#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#! pip install git+https://github.com/openai/whisper.git
#! pip install jiwer


# In[1]:


from avssl.data.flickr_dataset import FlickrDataset
import os
from tqdm import tqdm
from torch.nn.functional import normalize

flickr_base_path = "./data/flickr/"

dev_set = FlickrDataset(
    dataset_root=flickr_base_path,
    text_file='Flickr8k.token.txt',
    split="dev",
    load_image=True,
    modalities=["audio", "image", "text"],
)


# In[2]:


# very dumb coco loader which can be indexed by image index from locnarr dict
from PIL import Image
import util.localized_narratives
import torch

class CoCoDataset():
    
    def __init__(self, data_dir, split):
        self.data_dir = './' + data_dir
        self.split = split
        
    def __getitem__(self, index):
        fname = 'COCO_val2014_' + format(index, '012d') # could probably replace this with some sort of regex thing for robustness
        fpath = self.data_dir + '/' + self.split + '/' + fname + '.jpg'
        return Image.open(fpath)

class LocNarrDataset():
    def __init__(self, data_dir, split): # could be made less hard coded if we want to switch to supporting multiple image datasets
        self.coco_dataset = CoCoDataset(data_dir + '/coco/mscoco_img', split)
        self.locnarr_annot_loader = util.localized_narratives.DataLoader(data_dir + '/locnarr')
        self.locnarr_annot_loader.download_annotations('coco_val')
        self.annots = self.locnarr_annot_loader.load_annotations('coco_val')
    
    def generate(self):
        for annot in self.annots:
            to_return = dict()
            to_return['text'] = annot.caption
            to_return['wav'] = torch.tensor(self.locnarr_annot_loader.get_audioclip(annot.voice_recording))
            to_return['image'] = self.coco_dataset[int(annot.image_id)]
            to_return['id'] = int(annot.image_id)
            yield to_return
        
    



# In[4]:


coco_split = 'val2014'
coco_dir = f'./data/'

locnarr_dataset = LocNarrDataset(coco_dir, coco_split)


# In[5]:


import clip
import torch
import torchvision
model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()

import torch
import pandas as pd
import whisper
import torchaudio
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
whisper_model = whisper.load_model("base.en").eval()
options = whisper.DecodingOptions(language="en", without_timestamps=True)
print()


# In[ ]:


# can avoid running this if results are cached
def wav_to_text(wav):
    audio = whisper.pad_or_trim(wav.flatten()).cuda()
    mel = whisper.log_mel_spectrogram(audio)
    results = whisper_model.decode(mel, options).text
    return results

def text_to_embed(text):
    text_tokens = clip.tokenize(texts=text, truncate=True).cuda()
    with torch.no_grad():
        text_features = normalize(model.encode_text(text_tokens).float())
    return text_features

def img_to_embed(img):
    img_tensor = preprocess(img).cuda()[None,:,:,:]
    with torch.no_grad():
        image_features = normalize(model.encode_image(img_tensor).float())
    return image_features

import matplotlib.pyplot as plt
import numpy as np
image_feat_list = []
text_feat_list = []
whisper_text_feat_list = []
ids_seen = set()
whisper_texts = []
true_texts = []
img_to_text_idx = defaultdict(lambda : [])
text_to_img_idx = defaultdict(lambda : [])
id_to_img_idx = dict()



for idx, sample in tqdm(enumerate(locnarr_dataset.generate())):
    text = sample['text']
    img = sample['image']
    wav = sample['wav']
    whisper_text = wav_to_text(wav)
    text_features = text_to_embed(text)
    whisper_text_features = text_to_embed(whisper_text)
    image_features = img_to_embed(img)
    
    
    text_feat_list  += [text_features[0].tolist()]
    whisper_text_feat_list += [whisper_text_features[0].tolist()]
    whisper_texts += [whisper_text]
    true_texts += [text]
    
    if sample['id'] not in ids_seen:
        image_feat_list += image_features.tolist()
        id_to_img_idx[sample['id']] = len(image_feat_list)-1
        ids_seen.add(sample['id'])
    
    cur_image_idx = id_to_img_idx[sample['id']]
    cur_text_idx = len(true_texts)-1
    
    img_to_text_idx[cur_text_idx] += [cur_image_idx]
    text_to_img_idx[cur_image_idx] += [cur_text_idx]
    if idx == 1000:
        break
    
    
results = (image_feat_list, text_feat_list, whisper_text_feat_list, ids_list, whisper_texts, true_texts,
          dict(img_to_text_idx), dict(text_to_img_idx), id_to_img_idx)
with open('locnarr_whisper_results_cache.pkl', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


with open('whisper_results_cache.pkl', 'rb') as handle:
    results = pickle.load(handle)
image_feat_list, text_feat_list, whisper_text_feat_list, ids_list, whisper_texts, true_texts,\
          img_to_text_idx, text_to_img_idx, id_to_img_idx = results


# In[ ]:





# In[ ]:


from torchmetrics.functional import word_error_rate
whisper_lower = [text.lower() for text in whisper_texts]
true_lower = [text.lower() for text in true_texts]
WERs = [float(word_error_rate(whisper_lower[i], true_lower[i])) for i in range(len(whisper_lower))]
overall_WER = word_error_rate(whisper_lower, true_lower)


# In[ ]:


from sklearn.neighbors import KDTree
def get_recall(source_feat_list, target_feat_list, target_idx_to_source_idx_dict, recall_levels=[1, 5, 10]):
    # source_feat_list is the list of embeddings in the query modality
    # target_feat_list is the list of embeddings in the target modality
    # target_idx_to_source_idx_dict is a dictionary going from an index in target_feat_list to a list of indeces in source_feat_list
    # used to determine if a given target is correct for a given source
    tree = KDTree(target_feat_list)
    idxs = np.array(tree.query(source_feat_list, k=recall_levels[-1])[1])
    closest_ids = np.array([[target_idx_to_source_idx_dict[id] for id in idx] for idx in idxs])
    recall_avgs = []
    recalls = []
    for recall_level in recall_levels:
        recall_at_level = [i in np.array(closest_ids[i][0:recall_level]).flatten() for i in range(len(source_feat_list))]
        print(f'recall at level {recall_level}: {np.mean(recall_at_level)}')
        recall_avgs += [np.mean(recall_at_level)]
        recalls += [recall_at_level]
    return recalls, recall_avgs

print('CLIP base:')
print('text to image recall')
a = get_recall(text_feat_list, image_feat_list, text_to_img_idx)
print('image to text recall')
a = get_recall(image_feat_list, text_feat_list, img_to_text_idx)


# In[ ]:


from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

print('CLIP with Whisper:')
print('text to image recall')
a = get_recall(whisper_text_feat_list, image_feat_list, text_to_img_idx)
recall_at_5 = a[0][1]
print('image to text recall')
a = get_recall(image_feat_list, whisper_text_feat_list, img_to_text_idx)

plt.scatter(WERs, recall_at_5)
x_new = np.linspace(0, 1, 50)
logreg = LogisticRegression()
logreg_x = np.array(WERs)[:,np.newaxis]
logreg_y = np.array(recall_at_5).astype(int)
logreg.fit(logreg_x, logreg_y)
y_new = logreg.predict_proba(x_new[:,None])
plt.plot(x_new, y_new[:,1])
plt.xlabel('Per-sample WER')
plt.ylabel('Per-sample recall')
plt.title('Speech to Image R@5 vs. per-sample WER, Whisper+CLIP')
plt.show()


# In[ ]:


wav = dev_set[0]['wav']
audio = whisper.pad_or_trim(wav.flatten()).cuda()
mel = whisper.log_mel_spectrogram(audio)
results = model.decode(mel, options)
print(results.text)
print(dev_set[0]['text'])

