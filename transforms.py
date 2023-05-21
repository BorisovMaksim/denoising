import torch

from torchaudio.transforms import Resample
from torchvision.transforms import RandomCrop



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
        self.resample = Resample(orig_freq=input_sample_rate, new_freq=sample_rate)
        self.random_crop = RandomCrop((1, int(max_seconds * sample_rate)), pad_if_needed=True)
        self.normalize = normalize

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.input_sample_rate != self.sample_rate:
            waveform = self.resample(waveform)
        if self.normalize:
            waveform = waveform / torch.std(waveform)
        cropped = self.random_crop(waveform)
        return cropped
