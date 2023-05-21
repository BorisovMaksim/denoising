import noisereduce as nr
import torch
import torchaudio


class SpectralGating(torch.nn.Module):
    def __init__(self, rate=48000):
        super(SpectralGating, self).__init__()
        self.rate = rate

    def forward(self, wav):
        reduced_noise = torch.Tensor(nr.reduce_noise(y=wav, sr=self.rate))
        return reduced_noise
    
    def predict(self, wav):
        reduced_noise = torch.Tensor(nr.reduce_noise(y=wav, sr=self.rate))
        return reduced_noise



    
    



