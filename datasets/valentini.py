import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
import numpy as np
from torchaudio.transforms import Resample

HIGH_RANDOM_SEED = 1000

class Valentini(Dataset):
    def __init__(self, dataset_path, val_fraction, transform=None, valid=False, *args, **kwargs):
        clean_path = Path(dataset_path) / 'clean_trainset_56spk_wav'
        noisy_path = Path(dataset_path) / 'noisy_trainset_56spk_wav'
        clean_wavs = list(clean_path.glob("*"))
        noisy_wavs = list(noisy_path.glob("*"))
        valid_threshold = int(len(clean_wavs) * val_fraction)
        if valid:
            self.clean_wavs = clean_wavs[:valid_threshold]
            self.noisy_wavs = noisy_wavs[:valid_threshold]
        else:
            self.clean_wavs = clean_wavs[valid_threshold:]
            self.noisy_wavs = noisy_wavs[valid_threshold:]

        assert len(self.clean_wavs) == len(self.noisy_wavs)

        self.transform = transform
        self.valid = valid

    def __len__(self):
        return len(self.clean_wavs)

    def __getitem__(self, idx):
        noisy_wav, noisy_sr = torchaudio.load(self.noisy_wavs[idx])
        clean_wav, clean_sr = torchaudio.load(self.clean_wavs[idx])

        if self.transform:
            random_seed = 0 if self.valid else torch.randint(HIGH_RANDOM_SEED, (1,))[0]
            torch.manual_seed(random_seed)

            noisy_wav = self.transform(noisy_wav)
            torch.manual_seed(random_seed)
            clean_wav = self.transform(clean_wav)
        return noisy_wav, clean_wav




