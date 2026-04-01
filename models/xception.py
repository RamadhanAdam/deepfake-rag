import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution: spatial filtering per channel, then channel mixing."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    """One residual block: main path (SeparableConvs + MaxPool) + shortcut (1x1 conv)."""
    def __init__(self, in_channels, out_channels, reps, stride=1, start_with_relu=True, use_pooling=True):
        super(XceptionBlock, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(reps):
            in_c = in_channels if i == 0 else out_channels
            self.layers.append(nn.ReLU() if (start_with_relu or i > 0) else nn.Identity())
            self.layers.append(SeparableConv2d(in_c, out_channels, stride=1))
            self.layers.append(nn.BatchNorm2d(out_channels))

        if use_pooling:
            self.layers.append(nn.MaxPool2d(kernel_size=3, stride=stride, padding=1))

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = self.shortcut(x)
        for layer in self.layers:
            x = layer(x)
        return x + residual


class Xception(nn.Module):
    """
    Xception architecture built from scratch.
    Input:  (batch, 3, 299, 299)
    Output: (batch, num_classes)
    """
    def __init__(self, num_classes=2):
        super(Xception, self).__init__()

        # Entry flow - two regular convs then three residual blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU()

        self.block1 = XceptionBlock(64,  128, reps=2, stride=2, start_with_relu=False, use_pooling=True)
        self.block2 = XceptionBlock(128, 256, reps=2, stride=2, start_with_relu=True,  use_pooling=True)
        self.block3 = XceptionBlock(256, 728, reps=2, stride=2, start_with_relu=True,  use_pooling=True)

        # Middle flow - same block repeated 8 times, shape unchanged
        self.middle_flow = nn.Sequential(
            *[XceptionBlock(728, 728, reps=3, stride=1, start_with_relu=True, use_pooling=False)
              for _ in range(8)]
        )

        # Exit flow
        self.block4  = XceptionBlock(728, 1024, reps=2, stride=2, start_with_relu=True, use_pooling=True)
        self.sepconv1 = SeparableConv2d(1024, 1536)
        self.bn3      = nn.BatchNorm2d(1536)
        self.sepconv2 = SeparableConv2d(1536, 2048)
        self.bn4      = nn.BatchNorm2d(2048)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Entry flow
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # Middle flow
        x = self.middle_flow(x)
        # Exit flow
        x = self.block4(x)
        x = self.relu(self.bn3(self.sepconv1(x)))
        x = self.relu(self.bn4(self.sepconv2(x)))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x