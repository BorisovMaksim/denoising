import torch
from torchaudio.transforms import Resample
from torchvision.transforms import RandomCrop
from typing import Tuple


class Transform(torch.nn.Module):
    def __init__(
            self,
            input_sample_rate,
            sample_rate,
            max_seconds,
            normalize,
            *args,
            **kwargs
    ):
        super().__init__()
        self.input_sample_rate = input_sample_rate
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.resample = Resample(orig_freq=input_sample_rate, new_freq=sample_rate)
        # self.random_crop = RandomCrop((1, int(max_seconds * sample_rate)), pad_if_needed=True)
        self.normalize = normalize
        self.eps = 1e-7

    def random_crop_pad(self, clean, noisy):
        assert clean.shape[1] == noisy.shape[1]
        wav_length = clean.shape[1]
        target_length = self.sample_rate * self.max_seconds
        if wav_length > target_length:
            random_start = torch.randint(0, wav_length - target_length, (1,))[0]
            clean_resized = clean[:, random_start:random_start + target_length]
            noisy_resized = noisy[:, random_start:random_start + target_length]
        else:
            clean_resized = torch.concat((clean, torch.zeros((1, target_length - wav_length))), dim=1)
            noisy_resized = torch.concat((noisy, torch.zeros((1, target_length - wav_length))), dim=1)
        assert clean_resized.shape[1] == noisy_resized.shape[1]
        return clean_resized, noisy_resized

    def forward(self, clean_waveform: torch.Tensor, noisy_waveform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.input_sample_rate != self.sample_rate:
            clean_waveform = self.resample(clean_waveform)
            noisy_waveform = self.resample(noisy_waveform)
        if self.normalize:
            clean_waveform = clean_waveform / torch.std(clean_waveform)
            noisy_waveform = noisy_waveform / torch.std(noisy_waveform)

        clean_waveform, noisy_waveform = self.random_crop_pad(clean_waveform, noisy_waveform)
        return clean_waveform, noisy_waveform
