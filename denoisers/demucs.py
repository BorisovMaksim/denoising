import torch
from torch.nn.functional import pad
from utils import pad_cut_batch_audio
import torch.nn as nn


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, cfg):
        super(Encoder, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=cfg['conv1']['kernel_size'],
                                     stride=cfg['conv1']['stride'])
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(in_channels=out_channels, out_channels=2 * out_channels,
                                     kernel_size=cfg['conv2']['kernel_size'],
                                     stride=cfg['conv2']['stride'])
        self.glu = torch.nn.GLU(dim=-2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        if x.shape[-1] % 2 == 1:
            x = pad(x, (0, 1))
        x = self.glu(self.conv2(x))
        return x


class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, cfg, is_last=False):
        super(Decoder, self).__init__()
        self.is_last = is_last
        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=2 * in_channels,
                                     kernel_size=cfg['conv1']['kernel_size'],
                                     stride=cfg['conv1']['stride'])
        self.glu = torch.nn.GLU(dim=-2)
        self.conv2 = torch.nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=cfg['conv2']['kernel_size'],
                                              stride=cfg['conv2']['stride'])
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.glu(self.conv1(x))
        x = self.conv2(x)
        if not self.is_last:
            x = self.relu(x)
        return x


class Demucs(torch.nn.Module):
    def __init__(self, cfg):
        super(Demucs, self).__init__()
        self.L = cfg['L']

        encoders = [Encoder(in_channels=1, out_channels=cfg['H'], cfg=cfg['encoder'])]
        decoders = [Decoder(in_channels=cfg['H'], out_channels=1, cfg=cfg['decoder'], is_last=True)]
        for i in range(self.L - 1):
            encoders.append(Encoder(in_channels=(2 ** i) * cfg['H'],
                                    out_channels=(2 ** (i + 1)) * cfg['H'],
                                    cfg=cfg['encoder']))
            decoders.append(Decoder(in_channels=(2 ** (i + 1)) * cfg['H'],
                                    out_channels=(2 ** i) * cfg['H'],
                                    cfg=cfg['decoder']))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)

        self.lstm = torch.nn.LSTM(
            input_size=(2 ** (self.L - 1)) * cfg['H'],
            hidden_size=(2 ** (self.L - 1)) * cfg['H'], num_layers=2, batch_first=True)

    def forward(self, x):
        outs = [x]
        for i in range(self.L):
            out = self.encoders[i](outs[-1])
            outs.append(out)
        model_input = outs.pop(0)

        x, _ = self.lstm(outs[-1].permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        for i in reversed(range(self.L)):
            decoder = self.decoders[i]
            x = pad_cut_batch_audio(x, outs[i].shape)
            x = decoder(x + outs[i])
        x = pad_cut_batch_audio(x, model_input.shape)
        return x

    def predict(self, wav):
        prediction = self.forward(torch.reshape(wav, (1, 1, -1)))
        return prediction.detach()[0]

