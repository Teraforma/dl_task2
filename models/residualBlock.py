import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    please do not activate input, before putting in into this function
    """
    def __init__(self, channels):
        super.__init__()

        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels, channels/4, 1),
            nn.BatchNorm2d(channels/4),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels/4, channels/4, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(channels/4, channels, 1)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.block(x)

        out = out + identity

        out = self.relu(out)

        return out
