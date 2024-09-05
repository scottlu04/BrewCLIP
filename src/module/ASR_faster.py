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
import faster_whisper
import ctranslate2
from datasets import load_dataset, Audio,Dataset
from torch import nn
from torch.nn.utils.rnn import pad_sequence
#from transformers import pipeline
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator
from transformers.pipelines.pt_utils import KeyDataset
from .audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram

_whisper_models = {
    "tiny.en",
    "base.en",
    "small.en",
    "medium.en",
    "large.en",

}

class WhisperModel(faster_whisper.WhisperModel):
    '''
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    '''

    def generate_segment_batched(self, features: np.ndarray, tokenizer: faster_whisper.tokenizer.Tokenizer, options: faster_whisper.transcribe.TranscriptionOptions, encoder_output = None):
        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )

        encoder_output = self.encode(features)

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )

        result = self.model.generate(
                encoder_output,
                [prompt] * batch_size,
                length_penalty=options.length_penalty,
                max_length=self.max_length,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
            )

        tokens_batch = [x.sequences_ids[0] for x in result]

        def decode_batch(tokens: List[List[int]]) -> str:
            res = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            # text_tokens = [token for token in tokens if token < self.eot]
            return tokenizer.tokenizer.decode_batch(res)

        text = decode_batch(tokens_batch)

        return text

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        features = faster_whisper.transcribe.get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)



class FasterWhisperPipeline(Pipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """
    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def __init__(
            self,
            model,
            vad,
            options,
            tokenizer=None,
            device: Union[int, str, "torch.device"] = -1,
            framework = "pt",
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device

        super(Pipeline, self).__init__()
        self.vad_model = vad

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, audio):
        audio = audio['inputs']
        features = log_mel_spectrogram(audio, padding=N_SAMPLES - audio.shape[0])
        return {'inputs': features}

    def _forward(self, model_inputs):
        outputs = self.model.generate_segment_batched(model_inputs['inputs'], self.tokenizer, self.options)
        return {'text': outputs}

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor

        def stack(items):
            return {'inputs': torch.stack([x['inputs'] for x in items])}
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def transcribe(
        self, audio: Union[str, np.ndarray], batch_size=None, num_workers=0, language=None, task=None
    ) -> TranscriptionResult:
        if isinstance(audio, str):
            audio = load_audio(audio)

        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                # print(f2-f1)
                yield {'inputs': audio[f1:f2]}

        if self.tokenizer is None:
            language = language or self.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = faster_whisper.tokenizer.Tokenizer(self.model.hf_tokenizer,
                                                                self.model.model.is_multilingual, task=task,
                                                                language=language)
        else:
            language = language or self.tokenizer.language_code
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code:
                self.tokenizer = faster_whisper.tokenizer.Tokenizer(self.model.hf_tokenizer,
                                                                    self.model.model.is_multilingual, task=task,
                                                                    language=language)

        segments: List[SingleSegment] = []
        batch_size = batch_size or self._batch_size
        for idx, out in enumerate(self.__call__(data(audio, vad_segments), batch_size=batch_size, num_workers=num_workers)):
            text = out['text']
            if batch_size in [0, 1, None]:
                text = text[0]
            segments.append(
                {
                    "text": text,
                    "start": round(vad_segments[idx]['start'], 3),
                    "end": round(vad_segments[idx]['end'], 3)
                }
            )

        return {"segments": segments, "language": language}


    def detect_language(self, audio: np.ndarray):
        if audio.shape[0] < N_SAMPLES:
            print("Warning: audio is shorter than 30s, language detection may be inaccurate.")
        segment = log_mel_spectrogram(audio[: N_SAMPLES],
                                      padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
        encoder_output = self.model.encode(segment)
        results = self.model.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language = language_token[2:-2]
        print(f"Detected language: {language} ({language_probability:.2f}) in first 30s of audio...")
        return language
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
    model = WhisperModel(whisper_arch,
                         device=device,
                         device_index=device_index,
                         compute_type="float16",
                         download_root=download_root)
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