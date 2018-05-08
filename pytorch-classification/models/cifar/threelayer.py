'''A simple 3 layer network for CBAS-34.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['threelayer']

class ThreeLayer(nn.Module):

    def __init__(self, num_classes=34):
        super(ThreeLayer, self).__init__()
        self.convlayers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.lin = nn.Linear(4096, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.convlayers(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)
        x = self.classifier(x)
        return x

def threelayer(**kwargs):
    model = ThreeLayer(**kwargs)
    return model
