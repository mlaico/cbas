'''DileNet for CBAS-34.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


__all__ = ['dilenet']

class DileNet(nn.Module):

    def __init__(self, num_classes=34):
        super(DileNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def dilenet(**kwargs):
    """DileNet model architecture derived from AlexNet and dilated convolutions (Yu et al.)
    """
    model = DileNet(**kwargs)
    return model
