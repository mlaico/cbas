import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from cbas import CBAS
from torch.utils.data.sampler import WeightedRandomSampler as W_Sampler


def normal_weights(losses, mu=None):
    mu, var = mu if mu else np.mean(losses), np.var(losses)
    return (1/(np.sqrt(np.pi*2*var)))*np.exp(-((losses-mu)**2)/(2*var))


def size_train(training_set, model, loss_fn, optimizer, deviations):
    """
    training_set: class type 'torchvision.datasets.ImageFolder'

    deviations: a sequence of standard deviations scalars to be applied to the sampling distribution's
    mean to determine the probability of sampling and image with a given loss value. If set to [0...0],
    the probability of sampling each image (based on loss value) will be determined by the normal
    distribution's pdf. If deviation = -1, probability will be dictated by a normal with shifted mean
    mean(loss) -1*std(loss). This in effect allows us to shift the difficulty of training images over
    each epoch. Images are sampled with replacement, so we can shift the focus from easy to hard. For
    example: [-1, 0, 1] samples from a normal distribution centered at mean(loss) -1*std(loss),
    mean(loss), then mean(loss) + 1*std(loss) for the training epochs.

    Note: number of epochs == len(deviations) + 1 (+1 for the initial training epoch)
    """

    def size_curriculum(loader, net, criterion, optimizer):
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('%5d loss: %.3f' %
                      (i + 1, running_loss / 2000))
                running_loss = 0.0

    cbas_api = CBAS('annotations/cbas80.json')  # cbas splits coco train2017 into train and val
    training_ids = []
    ids = cbas_api.imIds.tolist()
    sizes = cbas_api.sizes

    for _, img in enumerate(training_set.imgs):  # here's were we get image ids from the torchvision dataset
        print(img)
        training_ids += img[0].split('/')[4].split('.')[0]
    cbas_ids = [i for i in range(len(ids)) if ids[i] in training_ids]
    ratios = sizes[cbas_ids]
    num_samples = ratios.shape[0]

    epoch = 0
    for deviation in deviations:
        epoch += 1
        print("epoch #%d" % epoch)
        weights = normal_weights(ratios, np.mean(ratios) + deviation*np.std(ratios))
        weights = weights / np.sum(weights)
        sampler = W_Sampler(weights, num_samples, replacement=True)
        size_curriculum_loader = \
            torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=False, sampler=sampler, num_workers=4)
        size_curriculum(size_curriculum_loader, model, loss_fn, optimizer)


