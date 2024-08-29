import torch
import torch.nn.functional as F
from torch import nn


# Define Late Fusion Classifier
class LFClassifier(nn.Module):
    def __init__(self, num_classes=2, num_clinical=7, kernel_size=3):
        super().__init__()
        self.conv_down_1 = DownSample(4, 16)
        self.conv_down_2 = DownSample(16, 32)
        self.conv_down_3 = DownSample(32, 64)

        self.fc1 = nn.Linear(64 * 52 * 52, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84 + num_clinical, 120)  # clinical features added
        self.fc4 = nn.Linear(120, 64 * 52 * 52)

        self.conv_bottleneck = DoubleConv(64, 128)

        self.conv_up_1 = UpSample(128, 64)
        self.conv_up_2 = UpSample(64, 32)
        self.conv_up_3 = UpSample(32, 16)

        self.conv_out = nn.Conv2d(16, num_classes, kernel_size, padding=1)

    def forward(self, x, clin):
        # Downsampling
        down_1, p1 = self.conv_down_1(x)
        down_2, p2 = self.conv_down_2(p1)
        down_3, p3 = self.conv_down_3(p2)

        # FC layers
        x = torch.flatten(p2, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.cat((x, clin), dim=1)
        x = self.fc3(x)
        x = self.fc4(x)
        x = x.view(x.shape[0], 64, 52, 52)

        x = self.conv_bottleneck(x)

        # Upsampling
        up_1 = self.conv_up_1(x, down_3)
        up_2 = self.conv_up_2(up_1, down_2)
        up_3 = self.conv_up_3(up_2, down_1)

        x = self.conv_out(up_3)
        x = F.softmax(x, dim=1)

        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)
