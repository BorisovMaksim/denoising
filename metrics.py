from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import torch
import torchaudio
from torchmetrics import SignalNoiseRatio


class Metrics:
    def __init__(self, rate=16000):
        self.nb_pesq = PerceptualEvaluationSpeechQuality(rate, 'wb')
        self.stoi = ShortTimeObjectiveIntelligibility(rate, False)
        self.snr = SignalNoiseRatio()
        
    def calculate(self, denoised, clean):
        return {'PESQ': self.nb_pesq(denoised, clean),
                'STOI': self.stoi(denoised, clean)}
    

