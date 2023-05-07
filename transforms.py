import torch

from torchaudio.transforms import Resample
from torchvision.transforms import RandomCrop



class Transform(torch.nn.Module):
    def __init__(
            self,
            input_sr,
            sample_rate,
            max_seconds,
            *args,
            **kwargs
    ):
        super().__init__()
        self.resample = Resample(orig_freq=input_sr, new_freq=sample_rate)
        self.random_crop = RandomCrop((1, int(max_seconds * sample_rate)), pad_if_needed=True)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        resampled = self.resample(waveform)
        croped = self.random_crop(resampled)
        return croped