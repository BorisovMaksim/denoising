import torch


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size_1=8, stride_1=4,
                 kernel_size_2=1, stride_2=1):
        super(Encoder, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                     kernel_size=kernel_size_1, stride=stride_1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(in_channels=out_channels, out_channels=2 * out_channels,
                                     kernel_size=kernel_size_2, stride=stride_2)
        self.glu = torch.nn.GLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.glu(self.conv2(x))
        return x


class Decoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size_1=3, stride_1=1,
                 kernel_size_2=8, stride_2=4):
        super(Decoder, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels=in_channels, out_channels=2 * in_channels,
                                     kernel_size=kernel_size_1, stride=stride_1)
        self.glu = torch.nn.GLU()
        self.conv2 = torch.nn.ConvTranspose1d(in_channels=in_channels, out_channels=out_channels,
                                              kernel_size=kernel_size_2, stride=stride_2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.glu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class Demucs(torch.nn.Module):
    def __init__(self):
        super(Demucs, self).__init__()

        self.encoder1 = Encoder(in_channels=1, out_channels=64)
        self.encoder2 = Encoder(in_channels=64, out_channels=128)
        self.encoder3 = Encoder(in_channels=128, out_channels=256)

        self.lstm = torch.nn.LSTM(input_size=256, hidden_size=256, num_layers=2)

        self.decoder1 = Decoder(in_channels=256, out_channels=128)
        self.decoder2 = Decoder(in_channels=128, out_channels=64)
        self.decoder3 = Decoder(in_channels=64, out_channels=1)

    def forward(self, x):
        out1 = self.encoder1(x)
        out2 = self.encoder2(out1)
        out3 = self.encoder3(out2)

        x = self.lstm(out3)

        x = self.decoder1(x + out3)
        x = self.decoder2(x + out2)
        x = self.decoder3(x + out1)

        return x
