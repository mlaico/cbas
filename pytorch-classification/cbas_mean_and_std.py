'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei
'''

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models


from utils import get_mean_and_std

transform_stats = transforms.Compose([
        transforms.ToTensor()
    ])

dataset = datasets.ImageFolder(root='../Data/coco/images/cbas34_train',transform=transform_stats)

cbas_mean, cbas_std = get_mean_and_std(dataset)

print('CBAS-34 mean: {}'.format(cbas_mean))
print('CBAS-34 std: {}'.format(cbas_std))
