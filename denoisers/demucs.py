import torch
from torch.nn.functional import pad

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=8, stride=2)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(in_channels=out_channels, out_channels=2 * out_channels,
                                     kernel_size=1, stride=1)
        self.glu = torch.nn.GLU(dim=-2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        if x.shape[-1] % 2 == 1:
            x = pad(x, (0, 1))
        x = self.glu(self.conv2(x))
        return x


class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=2 * in_channels,
                                     kernel_size=1, stride=1)
        self.glu = torch.nn.GLU(dim=-2)
        self.conv2 = torch.nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=8, stride=2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.glu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class Demucs(torch.nn.Module):
    def __init__(self, H):
        super(Demucs, self).__init__()

        self.encoder1 = Encoder(in_channels=1, out_channels=H)
        self.encoder2 = Encoder(in_channels=H, out_channels=2*H)
        self.encoder3 = Encoder(in_channels=2*H, out_channels=4*H)

        self.lstm = torch.nn.LSTM(
                                  input_size=4*H,
                                  hidden_size=4*H, num_layers=2, batch_first=True)

        self.decoder1 = Decoder(in_channels=4*H, out_channels=2*H)
        self.decoder2 = Decoder(in_channels=2*H, out_channels=H)
        self.decoder3 = Decoder(in_channels=H, out_channels=1)

    def forward(self, x):
        out1 = self.encoder1(x)
        out2 = self.encoder2(out1)
        out3 = self.encoder3(out2)

        x, _ = self.lstm(out3.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.decoder1(x + out3)
        x = x[:, :, :out2.shape[-1]]
        x = self.decoder2(x + out2)
        x = x[:, :, :-1]
        out1 = out1[:, :, :-1]
        if x.shape[-1] > out1.shape[-1]:
            x = x[:, :, :out1.shape[-1]]
        elif x.shape[-1] < out1.shape[-1]:
            out1 = out1[:, :, :x.shape[-1]]

        x = self.decoder3(x + out1)
        return x


