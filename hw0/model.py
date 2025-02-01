"""
This file is a copy from minzwon's repo sota-music-tagging-models/training/model.py, modules.py
(ref: https://github.com/minzwon/sota-music-tagging-models/blob/master/training)

Especially for the part of `ShortChunkCNN_Res` and related modules.
To better fit conditions of hw0, some changes are made
- n_fft 512 -> 2048
- hop_length -> 512
- the final activation layer Sigmoid() -> Softmax() for multi-class classification

-------------------------------------------------------
citation:
@inproceedings{won2020eval,
  title={Evaluation of CNN-based automatic music tagging models},
  author={Won, Minz and Ferraro, Andres and Bogdanov, Dmitry and Serra, Xavier},
  booktitle={Proc. of 17th Sound and Music Computing},
  year={2020}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.autograd import Variable

# DEVICE: GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Res_2d(nn.Module):
    def __init__(self, input_channels, output_channels, shape=3, stride=2):
        super(Res_2d, self).__init__()
        # convolution
        self.conv_1 = nn.Conv2d(
            input_channels, output_channels, shape, stride=stride, padding=shape // 2
        )
        self.bn_1 = nn.BatchNorm2d(output_channels)
        self.conv_2 = nn.Conv2d(
            output_channels, output_channels, shape, padding=shape // 2
        )
        self.bn_2 = nn.BatchNorm2d(output_channels)

        # residual
        self.diff = False
        if (stride != 1) or (input_channels != output_channels):
            self.conv_3 = nn.Conv2d(
                input_channels,
                output_channels,
                shape,
                stride=stride,
                padding=shape // 2,
            )
            self.bn_3 = nn.BatchNorm2d(output_channels)
            self.diff = True
        self.relu = nn.ReLU()

    def forward(self, x):
        # convolution
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))

        # residual
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.relu(out)
        return out


class ShortChunkCNN_Res(nn.Module):
    """
    Short-chunk CNN architecture with residual connections.
    """

    def __init__(
        self,
        n_channels=128,
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=128,
        n_class=11,
        with_meta=False,
    ):
        super(ShortChunkCNN_Res, self).__init__()
        self.n_channels = n_channels

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            f_min=f_min,
            f_max=f_max,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # CNN
        self.layer1 = Res_2d(1, n_channels, stride=2)
        self.layer2 = Res_2d(n_channels, n_channels, stride=2)
        self.layer3 = Res_2d(n_channels, n_channels * 2, stride=2)
        self.layer4 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer5 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer6 = Res_2d(n_channels * 2, n_channels * 2, stride=2)
        self.layer7 = Res_2d(n_channels * 2, n_channels * 4, stride=2)

        # Dense
        self.dense1 = nn.Linear(n_channels * 4, n_channels * 4)
        if with_meta:
            self.dense1 = nn.Linear(n_channels * 4 + 13, n_channels * 4)
        self.bn = nn.BatchNorm1d(n_channels * 4)
        self.dense2 = nn.Linear(n_channels * 4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x, meta=None):
        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        # CNN
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        # Global Max Pooling
        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)

        # Dense
        if meta is not None:
            x = torch.cat([x, meta], dim=1)
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Softmax(dim=-1)(x)

        return x
