from torch.utils.data import (
    DataLoader, Dataset
)
import logging
import tgt
import numpy as np
import os
import torch
import torchaudio
from tacotron2.tokenizer import BaseTokenizer
from tacotron2.utils import get_abspath

class TextMelProcessor:
    def __init__(self, cfg):
        super(TextMelProcessor, self).__init__()
        self.cfg = cfg

    def load_script(self, temp_path):
        raise NotImplementedError

    def get_dataset(self, split='train'):
        raise NotImplementedError


class TextMelDataset(Dataset):
    def __init__(self, cfg):
        super(TextMelDataset, self).__init__()
        self.cfg = cfg
        self.sampling_rate = cfg.sampling_rate
        self.filter_length = cfg.filter_length
        self.hop_length = cfg.hop_length
        self.win_length = cfg.win_length
        self.n_mel_channels = cfg.n_mel_channels
        self.mel_fmin = cfg.mel_fmin
        self.mel_fmax = cfg.mel_fmax
        self.train_script = cfg.train_script
        self.val_script = cfg.val_script
        self.load_mel_from_disk = cfg.load_mel_from_disk


    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def collater(self, batch):
        raise NotImplementedError

    def load_dataloader(self, shuffle: bool = True, batch_size: int = 2):
        return DataLoader(
            self, shuffle=shuffle, batch_size=batch_size, collate_fn=self.collater
        )


class TGTProcessor(TextMelProcessor):
    def __init__(self, cfg):
        super(TGTProcessor, self).__init__(cfg)

    def build_tokenizer(self):
        logging.info('start to build tokenizer')
        wav_text_scripts = get_abspath(self.cfg.train_script)
        wav_paths, transcripts = self.load_script(wav_text_scripts)
        tokenzier = BaseTokenizer.build_tokenizer(transcripts, self.cfg.normalize_option, self.cfg.g2p_lib)
        return tokenzier

    def load_script(self, temp_path):

        with open(temp_path, 'r') as f:
            lines = f.readlines()

        wav_paths = list()
        transcripts = list()

        for line in lines:
            line = line.strip().split("|")
            wav_paths.append(get_abspath(line[0]))
            transcripts.append(line[1])

        logging.info("load scripts from [{}]".format(temp_path))

        return wav_paths, transcripts

    def get_dataset(self, tokenizer, split='train'):
        if split=='train':
            wav_text_scripts = self.cfg.train_script
        else:
            wav_text_scripts = self.cfg.val_script
        wav_text_scripts = get_abspath(wav_text_scripts)
        wav_paths, transcripts = self.load_script(wav_text_scripts)

        text_infos = tokenizer.encode_batch(transcripts, add_special_token=False)['text_ids']

        logging.info("Convert raw lines to dataset")
        return TGTDataset(
            self.cfg, tokenizer, wav_paths, text_infos,
        )

class TGTDataset(TextMelDataset):
    def __init__(self, cfg, tokenizer, wav_paths, text_infos):
        super(TGTDataset, self).__init__(cfg)
        self.tokenizer = tokenizer
        self.wav_paths = wav_paths
        self.text_infos = text_infos

        self.pad_id = tokenizer.pad_id
        self.eos_id = tokenizer.eos_id
        self.bos_id = tokenizer.bos_id
        
        ## most vocoder use 'slaney' for mel scale.
        self.mel_converter = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sampling_rate,
            n_fft=cfg.filter_length,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            f_min=cfg.mel_fmin,
            f_max=cfg.mel_fmax,
            n_mels=cfg.n_mel_channels,
            power=1,
            #normalized=True,
            norm='slaney',
            mel_scale='slaney',
        )

        self.C = 1
        self.clip_val = 1e-5

    def get_mel(self, filename):
        dir, name = os.path.split(filename)
        cache_name = 'cache_{}'.format(name).replace('.wav', '.npy')
        cache_filename = os.path.join(dir, cache_name)

        load_flag = True
        melspec = None

        if self.load_mel_from_disk and os.path.exists(cache_filename):
            melspec = torch.from_numpy(np.load(cache_filename))
            if melspec.size(0) == self.n_mel_channels:
                load_flag = False

        if load_flag or melspec == None:
            ## torchaudio normalize audio to the interval [-1, 1]
            audio, sampling_rate = torchaudio.load(filename)

            ## only use mono
            if audio.size(0) > 1:
                audio = audio[0, :].view(1, -1)

            ## sample rate check
            if sampling_rate != self.sampling_rate:
                audio = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)(audio)

                ## Sometimes resmapling make problem to its boundary[-1, 1]
                ## So, add normalize function to scale audio
                if audio.max() > 1 or audio.min() < -1:
                    def normalize(tensor):
                        # Subtract the mean, and scale to the interval [-1,1]
                        tensor_minusmean = tensor - tensor.mean()
                        return tensor_minusmean / tensor_minusmean.abs().max()

                    audio = normalize(audio)

            try:
                melspec = self.mel_converter(audio)
                melspec = torch.log(torch.clamp(melspec, min=self.clip_val) * self.C)
            except Exception:
                raise ValueError("ERROR : {}".format(filename))
            melspec = torch.squeeze(melspec, 0)

            if self.load_mel_from_disk:
                np.save(cache_filename, melspec.numpy())

        return melspec

    def __getitem__(self, idx):
        text_ids = self.text_infos[idx]
        wav_path = self.wav_paths[idx]
        mel_specs = self.get_mel(wav_path)

        return {
            'text_ids': torch.tensor(text_ids, dtype=torch.long),
            'mel_specs': torch.tensor(mel_specs, dtype=torch.float),
        }

    def __len__(self):
        return len(self.wav_paths)

    def collater(self, batch):
        batch_size = len(batch)

        text_ids = [b['text_ids'] for b in batch]
        mel_specs = [b['mel_specs'] for b in batch]

        mel_lengths = [mel_spec.size(1) for mel_spec in mel_specs]
        target_mel_size = max(mel_lengths)

        collated_mel_specs = torch.zeros((batch_size, self.n_mel_channels, target_mel_size), dtype=torch.float)
        collated_gate_targets = torch.zeros((batch_size, target_mel_size), dtype=torch.float)

        for i, (mel_spec, mel_length) in enumerate(zip(mel_specs, mel_lengths)):
            diff = mel_length - target_mel_size
            if diff > 0:
                collated_mel_specs[i, :, :target_mel_size] = mel_spec[:, :target_mel_size]
                #mel_length = target_mel_size
                collated_gate_targets[i, target_mel_size-1] = 1.0

            else:
                collated_mel_specs[i, :, :mel_length] = mel_spec
                collated_gate_targets[i, mel_length-1] = 1.0


        text_lengths = [len(text_id) for text_id in text_ids]
        target_text_size = max(text_lengths)

        collated_ids =  torch.full((batch_size, target_text_size), self.pad_id, dtype=torch.long)
        
        for i, (text_id, text_length) in enumerate(zip(text_ids, text_lengths)):
            diff = text_length - target_text_size
            if diff > 0:
                collated_ids[i, :target_text_size] = text_id[:target_text_size]
                #text_length = target_text_size
            else:
                collated_ids[i, :text_length] = text_id


        return {
            'text_ids': collated_ids,
            'text_lengths': torch.tensor(text_lengths, dtype=torch.long),
            'mel_specs': collated_mel_specs,
            'mel_lengths': torch.tensor(mel_lengths, dtype=torch.long),
            'gate_targets' : collated_gate_targets,
        }


