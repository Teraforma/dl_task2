from urllib import request

import torch
import torch.nn as nn
from  models.residualBlock import ResidualBlock

class Alpha(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()

        self.features = nn.Sequential(
            # comments are my intentions behind the structure
            # extract as much as  possible raw element, here groups 3 is intensional,
            # cause I want it to be kind building blocks for coming layers
            ###save dimensions of each channel, but channels goes x32
            nn.Conv2d(3, 96, 3, padding=1, groups=3),
            nn.ReLU(),

            # building features from building blocks
            ### each channel dims are halved,channels: 96->512
            nn.Conv2d(96, 512, 3, stride=2, padding=1),


            # res blocks to process
            ### preserves channels and their size
            ResidualBlock(512),
            ResidualBlock(512),

            # subsample to decrees the size of each channel
            ### each channel size is halved, channels: same
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            #nn.Conv2d(512, 1024, 3, stride=2, padding=1),

            # because i use maxpool, output is the same if maxpool->relu or relu->maxpool

            ResidualBlock(512),
            ResidualBlock(512),

            # subsample again
            ### channel size is halved
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            ResidualBlock(512),
            ResidualBlock(512),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # average value of all windows by the channels
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = Alpha(37)
