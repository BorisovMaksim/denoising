import torch
from torch.utils.data import Dataset
from pathlib import Path
from utils import load_wav


class Valentini(Dataset):
    def __init__(self, dataset_path='/media/public/datasets/denoising/DS_10283_2791/', transform=None,
                 valid=False):
        clean_path = Path(dataset_path) / 'clean_trainset_56spk_wav'
        noisy_path = Path(dataset_path) / 'noisy_trainset_56spk_wav'
        clean_wavs = list(clean_path.glob("*"))
        noisy_wavs = list(noisy_path.glob("*"))
        valid_threshold = int(len(clean_wavs) * 0.2)
        if valid:
            self.clean_wavs = clean_wavs[:valid_threshold]
            self.noisy_wavs = noisy_wavs[:valid_threshold]
        else:
            self.clean_wavs = clean_wavs[valid_threshold:]
            self.noisy_wavs = noisy_wavs[valid_threshold:]

        assert len(self.clean_wavs) == len(self.noisy_wavs)

        self.transform = transform

    def __len__(self):
        return len(self.clean_wavs)

    def __getitem__(self, idx):
        noisy_wav = load_wav(self.noisy_wavs[idx])
        clean_wav = load_wav(self.clean_wavs[idx])

        if self.transform:
            random_seed = torch.randint(100, (1,))[0]
            torch.manual_seed(random_seed)
            noisy_wav = self.transform(noisy_wav)
            torch.manual_seed(random_seed)
            clean_wav = self.transform(clean_wav)
        return noisy_wav, clean_wav
