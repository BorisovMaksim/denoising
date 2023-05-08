import pesq
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
        pesq_scores, stoi_scores = 0, 0
        for denoised_wav, clean_wav in zip(denoised, clean):
            try:
                pesq_scores += self.nb_pesq(denoised_wav, clean_wav).item()
                stoi_scores += self.stoi(denoised_wav, clean_wav).item()
            except pesq.NoUtterancesError as e:
                print(e)
            except ValueError as e:
                print(e)


        return {'PESQ': pesq_scores,
                'STOI': stoi_scores}
    

