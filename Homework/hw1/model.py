import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, Wav2Vec2FeatureExtractor

# DEVICE: GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model
MERT_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)

# Freeze the pretrained model's parameters
for param in MERT_model.parameters():
    param.requires_grad = False


class CNNClassifier(nn.Module):
    """
    1D CNN-based Multi-label Classifier
    """

    def __init__(
        self,
        n_channels=128,
        n_class=9,
    ):
        super(CNNClassifier, self).__init__()
        self.n_channels = n_channels

        # CNN
        self.layer1 = nn.Conv1d(25, 1, kernel_size=1)
        self.layer2 = nn.Conv1d(n_channels, n_channels, stride=2, kernel_size=1)
        self.layer3 = nn.Conv1d(n_channels, n_channels * 2, stride=2, kernel_size=1)
        self.layer4 = nn.Conv1d(n_channels * 2, n_channels * 2, stride=2, kernel_size=1)
        self.layer5 = nn.Conv1d(n_channels * 2, n_channels * 2, stride=2, kernel_size=1)
        self.layer6 = nn.Conv1d(n_channels * 2, n_channels * 2, stride=2, kernel_size=1)

        # Dense
        self.dense1 = nn.Linear(1024, n_channels)
        self.bn = nn.BatchNorm1d(n_channels)
        self.dense2 = nn.Linear(n_channels, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # CNN
        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        # x = self.layer6(x)

        # Global Max Pooling
        if x.size(1) != 1:
            print("pooling")
            x = nn.MaxPool1d(x.size(1))(x)
        x = x.squeeze(1)

        # Dense
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x


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
        sample_rate=24000,
        n_fft=2048,
        hop_length=512,
        f_min=0.0,
        f_max=8000.0,
        n_mels=128,
        n_class=9,
    ):
        super(ShortChunkCNN_Res, self).__init__()
        self.n_channels = n_channels

        # # Pre-trained encoder
        # self.mert = MERT_model
        # for param in self.mert.parameters():
        #     param.requires_grad = False

        # Spectrogram
        self.conv1d = nn.Conv1d(374, 128, kernel_size=3, padding=1)
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
        self.bn = nn.BatchNorm1d(n_channels * 4)
        self.dense2 = nn.Linear(n_channels * 4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pre-trained Encoder
        # outputs = self.mert(**inputs, output_hidden_states=True)
        # x = outputs.last_hidden_state  # [batch_size, time, 1024 feature_dim]

        # Spectrogram
        x = self.conv1d(x)
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
        x = self.dense1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = nn.Sigmoid()(x)

        return x
