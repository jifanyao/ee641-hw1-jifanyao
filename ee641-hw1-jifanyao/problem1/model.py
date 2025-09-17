import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.pred = nn.Conv2d(in_channels, num_anchors * (5 + num_classes), 1)
    def forward(self, x):
        x = self.conv(x)
        return self.pred(x)

class MultiScaleDetector(nn.Module):
    def __init__(self, num_classes=3, num_anchors=3):
        """
        Multi-scale SSD-like detector.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Backbone
        self.block1a = ConvBlock(3, 32, 1)
        self.block1b = ConvBlock(32, 64, 2)  # [224->112]
        self.block2 = ConvBlock(64, 128, 2)  # [112->56]
        self.block3 = ConvBlock(128, 256, 2) # [56->28]
        self.block4 = ConvBlock(256, 512, 2) # [28->14]

        # Detection heads
        self.head1 = DetectionHead(128, num_anchors, num_classes)  # 56x56
        self.head2 = DetectionHead(256, num_anchors, num_classes)  # 28x28
        self.head3 = DetectionHead(512, num_anchors, num_classes)  # 14x14

    def forward(self, x):
        x = self.block1a(x)
        x = self.block1b(x)
        s1 = self.block2(x)
        s2 = self.block3(s1)
        s3 = self.block4(s2)

        p1 = self.head1(s1)
        p2 = self.head2(s2)
        p3 = self.head3(s3)

        return [p1, p2, p3]

