import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
import numpy as np
from torchaudio.transforms import Resample


class Minimal(Dataset):
    def __init__(self, cfg):
        self.wavs = ['p232_284.wav', 'p232_071.wav', 'p257_171.wav', 'from_train.wav']
        self.dataset_path = cfg['validation']['path']
        self.target_rate = cfg['dataloader']['sample_rate']
        self.resampler = Resample(orig_freq=cfg['validation']['sample_rate'],
                                  new_freq=cfg['dataloader']['sample_rate'])

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        wav, rate = torchaudio.load(Path(self.dataset_path) / self.wavs[idx])
        wav = self.resampler(wav)
        return wav, self.target_rate
