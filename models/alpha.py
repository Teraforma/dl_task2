import torch
import torch.nn as nn
from  residualBlock import ResidualBlock

class Alpha(nn.Module):
    def __init__(self, num_classes=37):
        super(Alpha, self).__init__()

        self.features = nn.Sequential(
            # comments are my intentions behind the structure
            # extract as much as  possible raw element, here groups 3 is intensional,
            # cause I want it to be kind building blocks for coming layers
            nn.Conv2d(3, 96, 3, padding=1, groups=3),
            nn.ReLU(),

            # building features from building blocks
            nn.Conv2d(96, 512, 3, stride=2),


            # res blocks to process
            ResidualBlock(512),
            ResidualBlock(512),

            # subsample to decrees the size of each channel
            #nn.Conv2d(512, 1024, 3, stride=2),
            nn.MaxPool2d(2,2),
            nn.ReLU(),# because i use maxpool, output is the same if maxpool->relu or relu->maxpool

            ResidualBlock(1024),
            ResidualBlock(1024),

            # subsample again
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            ResidualBlock(2048),
            ResidualBlock(2048),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)