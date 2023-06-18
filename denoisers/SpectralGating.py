import noisereduce as nr
import torch
import torchaudio


class SpectralGating(torch.nn.Module):
    def __init__(self, rate=16000):
        super(SpectralGating, self).__init__()
        self.rate = rate

    def forward(self, wav):
        reduced_noise = torch.Tensor(nr.reduce_noise(y=wav, sr=self.rate))
        return reduced_noise
    
    def predict(self, wav_path, out_path):
        data, rate = torchaudio.load(wav_path)
        reduced_noise = torch.Tensor(nr.reduce_noise(y=data, sr=rate))
        torchaudio.save(out_path, reduced_noise, rate)
        return out_path



    
    



