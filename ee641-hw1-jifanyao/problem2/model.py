import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2)
        )  # 128→64
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)
        )  # 64→32
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2)
        )  # 32→16
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2)
        )  # 16→8

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return c1, c2, c3, c4


class HeatmapNet(nn.Module):
    def __init__(self, num_keypoints=5):
        super().__init__()
        self.encoder = Encoder()
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.BatchNorm2d(128), nn.ReLU()
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, stride=2), nn.BatchNorm2d(64), nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 2, stride=2), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.final = nn.Conv2d(32, num_keypoints, 1)

    def forward(self, x):
        c1, c2, c3, c4 = self.encoder(x)
        d4 = self.deconv4(c4)
        d3 = self.deconv3(torch.cat([d4, c3], dim=1))
        d2 = self.deconv2(torch.cat([d3, c2], dim=1))
        out = self.final(d2)
        return out


class RegressionNet(nn.Module):
    def __init__(self, num_keypoints=5):
        super().__init__()
        self.encoder = Encoder()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_keypoints * 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        _, _, _, c4 = self.encoder(x)
        x = self.gap(c4).flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        coords = torch.sigmoid(self.fc3(x))
        return coords
