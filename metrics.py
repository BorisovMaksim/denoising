from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
import torch
import torchaudio



class Metrics:
    def __init__(self, rate=16000):
        self.nb_pesq = PerceptualEvaluationSpeechQuality(rate, 'wb')
        self.stoi = ShortTimeObjectiveIntelligibility(rate, False)
        
    def calculate(self, preds, target):
        return {'PESQ': self.nb_pesq(preds, target),
                'STOI': self.stoi(preds, target)}
    

