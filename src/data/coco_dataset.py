import json
import logging

logger = logging.getLogger(__name__)
import os
import re
from collections import defaultdict
from typing import List

import clip

from .base_dataset import BaseDataset


class COCODataset(BaseDataset):
    def __init__(
        self,
        dataset_root: str,
        modalities: List,
        split: str = "train",
        image_transform=None,
        audio_transform=None,
        target_sr: int = 16_000,
        load_audio: bool = True,
        load_image: bool = True,
        wav_rm_silence: bool = False,
        clip_image_transform: str = None,
        split_prefix: str = "SpokenCOCO",
        **kwargs,
    ):
        if clip_image_transform is not None:
            logger.info(
                "Load clip ({}) for image transform".format(clip_image_transform)
            )
            _, image_transform = clip.load(clip_image_transform, "cpu")
        super().__init__(
            dataset_root=dataset_root,
            split=split,
            image_transform=image_transform,
            audio_transform=audio_transform,
            target_sr=target_sr,
            load_audio=load_audio,
            load_image=load_image,
            **kwargs,
        )

        assert len(modalities) > 0, "Dataset's modalities cannot be none"
        self.modalities = modalities

        assert self.split in ["train", "val", "test"]

        data_json_path = os.path.join(
            self.dataset_root, "SpokenCOCO", f"{split_prefix}_{self.split}.json"
        )
        logger.info(f"data_json_path {data_json_path}")
        with open(data_json_path, "r") as f:
            raw_data = json.load(f)["data"]

        for _entry in raw_data:
            if "audio" in self.modalities or "text" in self.modalities:
                data_id = (
                    _entry["reassign_id"]
                    if split_prefix != "SpokenCOCO"
                    else int(_entry["image"].split("_")[-1].replace(".jpg", ""))
                )
                for _capion in _entry["captions"]:
                    _ent_data = {
                        "id": data_id,
                    }

                    if "audio" in self.modalities:
                        _ent_data["wav"] = os.path.join(
                            self.dataset_root, "SpokenCOCO", _capion["wav"]
                        )
                    if "image" in self.modalities:
                        _ent_data["image"] = os.path.join(
                            self.dataset_root, "mscoco_img", _entry["image"]
                        )
                    if "text" in self.modalities:
                        _ent_data["text"] = _capion["text"].lower()
                    self.data.append(_ent_data)
            else:
                self.data.append(
                    {
                        "image": os.path.join(
                            self.dataset_root, "mscoco_img", _entry["image"]
                        ),
                        "id": data_id,
                    }
                )

        logger.info(f"SpokenCOCO ({self.split}): {len(self.data)} samples")
    def add_transcription_full(self, aud_transcription_file):
        aud2transcription = {}
        with open(os.path.join(self.dataset_root, aud_transcription_file), "r") as fp:
           # count = 0
            for line in fp:
                #count +=1
                #print(count)
                out = line.split('# ')
                if out[0] == "2699733386_c346c87ea6_0":
                    print(out)
                if len(out) == 2:
                    audio_name, transcription = out
                elif len(out) == 1:
                    continue
                else:
                    ValueError
                #print(audio_name)
                #print(transcription)
                aud2transcription[audio_name] = transcription
        missing_count = 0
        new_data = []
        for ind, x in enumerate(self.data):
            #print(x['wav'])
            ##path = os.path.join(self.dataset_root,'flickr_audio',"wavs", x['wav'] +'.wav')
            name  = x['wav'].split("/")[-1][:-4]
            #print(name)
            if name == "2699733386_c346c87ea6_0":
                print(x)
            #print(path)
            if name in aud2transcription:
                #x.update( {"transcription":aud2transcription[name]})
                x['transcription'] = aud2transcription[name]
                new_data.append(x)
                #continue
            else:
                # del self.data[ind]
                missing_count +=1
                print("Missing" + name)
            # if name == "2699733386_c346c87ea6_0":
            #     print(x)
        print(f'Missing {missing_count} transcriptions.')
        self.data = new_data
        # for ind, x in enumerate(self.data):
        #     name  = x['wav'].split("/")[-1][:-4]
        #     #print(x)
        #     print(x['transcription'])
