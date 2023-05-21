import pesq
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchaudio.transforms import Resample
import torch
import torchaudio
from torchmetrics import SignalNoiseRatio

class Metrics(torch.nn.Module):
    def __init__(self, source_rate, target_rate=16000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_rate = source_rate
        self.target_rate = target_rate
        self.resampler = Resample(orig_freq=source_rate, new_freq=target_rate)
        self.nb_pesq = PerceptualEvaluationSpeechQuality(target_rate, 'wb')
        self.stoi = ShortTimeObjectiveIntelligibility(target_rate, False)
        self.snr = SignalNoiseRatio()
        
    def forward(self, denoised, clean):
        pesq_scores, stoi_scores = 0, 0
        for denoised_wav, clean_wav in zip(denoised, clean):
            if self.source_rate != self.target_rate:
                denoised_wav = self.resampler(denoised_wav)
                clean_wav = self.resampler(clean_wav)
            try:
                pesq_scores += self.nb_pesq(denoised_wav, clean_wav).item()
                stoi_scores += self.stoi(denoised_wav, clean_wav).item()
            except pesq.NoUtterancesError as e:
                print(e)
            except ValueError as e:
                print(e)

        return {'PESQ': pesq_scores,
                'STOI': stoi_scores}
    

