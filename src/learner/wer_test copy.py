import numpy as np
import torch
from easydict import EasyDict as edict
import os, os.path as osp
import yaml
import sys 
import importlib 
import logging


from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader, random_split
if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))
from tqdm import tqdm

sys.path.append('..');
from model import *
from module import asr
from data import collate_general

dataset_dict = {
        'flickr8k' : 'Flickr8kDataset',
        'coco' : 'COCODataset',
        'ln_coco' : 'LN_COCO_Dataset',
        'ln_flickr30k' : 'LN_Flickr30k_Dataset',
        'ravdess' : 'RAVDESS_Dataset',
    }

class Wer_test(object):

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        seed_everything(self.args.seed)
        
        print("combined cfg:")
        self.cfg.update(edict(vars(self.args)))
        print(self.cfg)
        # if self.cfg.mode == 'pipeline':
        #     self.model = pipeline(self.cfg)
        # elif self.cfg.mode == 'e2e':
        #     self.model = e2e(self.cfg)
        #self.model = brewclip(self.cfg)

        path = self.cfg.pretrained_path
        checkpoint = torch.load(path)
        self.brewclip = brewclip(self.cfg).cuda()
        self.brewclip.load_state_dict(checkpoint["state_dict"])

        self.dataset_name = self.cfg.data.dataset.name
        assert self.dataset_name in dataset_dict
        

    def test(self):
        test_set = getattr(importlib.import_module('.'+ self.dataset_name +'_dataset', package='data'), 
                    dataset_dict[self.dataset_name])(split="test", **self.cfg.data.dataset,)
        test_set.add_transcription_full('aud2trans_test_'+self.dataset_name+'.txt')
        # image_feat_list = []
        # text_feat_list = []
        # whisper_text_feat_list = []
        # ids_seen = set()
        # whisper_texts = []
        # true_texts = []
        # img_to_text_idx = defaultdict(lambda : [])
        # text_to_img_idx = defaultdict(lambda : [])
        # id_to_img_idx = dict()
        # for idx, sample in tqdm(enumerate(test_loader)): #could also tqdm over dev_set
        #     text = sample['text']
        #     img = sample['image']
        #     wav = sample['wav']
        #     whisper_text = wav_to_text(wav, whisper_model, options)
        #     text_features = text_to_embed(text, model)
        #     whisper_text_features = text_to_embed(whisper_text, model)
        #     image_features = img_to_embed(img, model, preprocess)
            
            
        #     # text_feat_list  += [text_features[0].tolist()]
        #     # whisper_text_feat_list += [whisper_text_features[0].tolist()]
        #     # whisper_texts += [whisper_text]
        #     # true_texts += [text]
            
        #     if sample['id'] not in ids_seen:
        #         image_feat_list += image_features.tolist()
        #         id_to_img_idx[sample['id']] = len(image_feat_list)-1
        #         ids_seen.add(sample['id'])
            
        #     cur_image_idx = id_to_img_idx[sample['id']]
        #     cur_text_idx = len(true_texts)-1
            
        #     img_to_text_idx[cur_text_idx] += [cur_image_idx]
        #     text_to_img_idx[cur_image_idx] += [cur_text_idx]
        #     if idx == coco_samples_to_eval-1:
        #         break
        from sklearn.neighbors import KDTree
        def get_recall(source_feat_list, target_feat_list, target_idx_to_source_idx_dict, recall_levels=[1, 5, 10]):
            # source_feat_list is the list of embeddings in the query modality
            # target_feat_list is the list of embeddings in the target modality
            # target_idx_to_source_idx_dict is a dictionary going from an index in target_feat_list to a list of indeces in source_feat_list
            # used to determine if a given target is correct for a given source
            tree = KDTree(target_feat_list)
            idxs = np.array(tree.query(source_feat_list, k=recall_levels[-1])[1])
            a = idxs[0]
            closest_ids = [[target_idx_to_source_idx_dict[id] for id in idx] for idx in idxs]
            recall_avgs = []
            recalls = []
            for recall_level in recall_levels: 
                recall_at_level = []
                for source_idx in range(len(source_feat_list)):
                    recalls_list = closest_ids[source_idx] # list of lists
                    recalls_list = recalls_list[0:recall_level] # shortened to only contain ones for current recall level
                    recalls_list_flat = []
                    for a in recalls_list:
                        recalls_list_flat += a
                    recall_at_level += [source_idx in recalls_list_flat]
                print(f'recall at level {recall_level}: {np.mean(recall_at_level)}')
                recall_avgs += [np.mean(recall_at_level)]
                recalls += [recall_at_level]
            return recalls, recall_avgs

        test_loader = DataLoader(
                test_set,
                batch_size=1,
                shuffle=False,
                num_workers=self.cfg.njobs,
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_general,
            )
        # bar = tqdm(total=len(test_loader), desc='train_data_transcribe steps', dynamic_ncols=False)
        # for idx, sample in tqdm(enumerate(test_loader)):
        #     text = sample['text']
        #     transcription = sample['transcription']
        #     wav = sample['wav']
        #     out = self.brewclip(sample)
        #     audio_feat = out['audio_feat']
        #     image_feat = out['image_feat']
        #     print(idx)
        #     bar.update()
        # bar.close()
        from collections import defaultdict
        full_out_list = []
        image_feat_list = []
        text_feat_list = []
        whisper_text_feat_list = []
        ids_seen = set()
        whisper_texts = []
        true_texts = []
        img_to_text_idx = defaultdict(lambda : [])
        text_to_img_idx = defaultdict(lambda : [])
        id_to_img_idx = dict()
        id_list = []
        coco_samples_to_eval = 10000000

        print(len(test_loader))
        for idx, sample in enumerate(test_loader): #could also tqdm over dev_set
            print(idx)
            text = sample['text']
            transcription = sample['transcription']
            import clip
            def _TokenizeText(texts: Union[str, List[str]]):
                return clip.tokenize(texts=texts, truncate=True)

            sample['text'] = _TokenizeText(sample['text'])
            sample['transcription'] = _TokenizeText(sample['transcription'])
            def tensor_dict_to_cuda(tdict: dict, gpu: int) :
                for k in tdict :
                    if isinstance(tdict[k], dict) :
                        tensor_dict_to_cuda(tdict[k], gpu)
                        continue
                    
                    if torch.is_tensor(tdict[k]) :
                        tdict[k] = tdict[k].cuda()#gpu, non_blocking=True) 
            tensor_dict_to_cuda(sample,0)
            out = self.brewclip(sample)
            #full_out_list +=[out]#.cpu().tolist()
            audio_feat = out['audio_feat']
            image_feat = out['image_feat']
            id_list += sample['id']
            #print(full_out_list)
            text_feat_list  += audio_feat.cpu().tolist()
           # whisper_text_feat_list += [whisper_text_features[0].tolist()]
            whisper_texts += transcription
            true_texts += text
            #print(text_feat_list)
            
            if sample['id'] not in ids_seen:
                image_feat_list += image_feat.cpu().tolist()
                id_to_img_idx[sample['id']] = len(image_feat_list)-1
                ids_seen.add(sample['id'])
            
            cur_image_idx = id_to_img_idx[sample['id']]
            cur_text_idx = len(true_texts)-1
            
            img_to_text_idx[cur_text_idx] += [cur_image_idx]
            text_to_img_idx[cur_image_idx] += [cur_text_idx]
            if idx == coco_samples_to_eval-1:
               break
            
        import pickle  
        results = (id_list, image_feat_list, text_feat_list, whisper_texts, true_texts,
                dict(img_to_text_idx), dict(text_to_img_idx), id_to_img_idx)
        with open('locnarr_whisper_results_cache2.pkl', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                # print('CLIP base, locnarr:')
                # print('text to image recall')
                # a = get_recall(text_feat_list, image_feat_list, text_to_img_idx)
                # print('image to text recall')
                # a = get_recall(image_feat_list, text_feat_list, img_to_text_idx)