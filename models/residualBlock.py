import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    please do not activate input, before putting in into this function
    """
    def __init__(self, channels):
        super().__init__()
        inner_channels = int (channels/4)

        self.block = nn.Sequential(
            # do not change channels
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            # channels size is not changed, but the number is 1/4
            nn.Conv2d(channels, inner_channels, 1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),

            # because of padding size is not changed
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),

            # channel size is the same, but number of channels goes x4
            nn.Conv2d(inner_channels, channels, 1)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.block(x)

        out = out + identity

        out = self.relu(out)

        return out
