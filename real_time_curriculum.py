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


class MySampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.random_list = None

    def __iter__(self):
        self.random_list = torch.randperm(len(self.data_source)).tolist()
        return iter(self.random_list)

    def get_idx(self):
        return self.random_list

    def __len__(self):
        return len(self.data_source)


class MyWeightedSampler(Sampler):

    def __init__(self, weights, num_samples, replacement=True):
        self.weights = torch.DoubleTensor(weights)
        self.num_samples = num_samples
        self.replacement = replacement
        self.random_list = None

    def __iter__(self):
        ret = torch.multinomial(self.weights, self.num_samples, self.replacement)
        self.random_list = ret.numpy().tolist()
        return iter(ret)

    def get_idx(self):
        return self.random_list

    def __len__(self):
        return self.num_samples


def normal_weights(losses, mu=None):
    mu, var = mu if mu else np.mean(losses), np.var(losses)
    return (1/(np.sqrt(np.pi*2*var)))*np.exp(-((losses-mu)**2)/(2*var))


def real_time(training_set, model, loss_fn, optimizer, deviations):
    """
    training_set: class type 'torchvision.datasets.ImageFolder'

    deviations: a sequence of standard deviations scalars to be applied to the sampling distribution's
    mean to determine the probability of sampling and image with a given loss value. If set to [0...0],
    the probability of sampling each image (based on loss value) will be determined by the normal
    distribution's pdf. If deviation = -1, probability will be dictated by a normal dist with shifted mean
    mean(loss) -1*std(loss). This in effect allows us to shift the difficulty of training images over
    each epoch. Images are sampled with replacement, so we can shift the focus from easy to hard. For
    example: [-1, 0, 1] samples from a normal distribution centered at mean(loss) -1*std(loss),
    mean(loss), then mean(loss) + 1*std(loss) for the training epochs.

    Note: number of epochs == len(deviations) + 1 (+1 for the initial training epoch)
    """

    def real_time_curriculum(sampler, loader, net, criterion, optimizer):
        orderings = []
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            # get the inputs
            inputs, labels = data

            try:
                numpy_labels = labels.numpy()
            except:
                numpy_labels = labels.data.numpy()

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            try:
                numpy_outputs = outputs.numpy()
            except:
                numpy_outputs = outputs.data.numpy()
            log_probs = -np.log(np.exp(numpy_outputs)
                                / np.reshape(np.sum(np.exp(numpy_outputs), axis=1), (numpy_labels.shape[0], 1)))
            orderings += log_probs[:, numpy_labels].tolist()[0]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('%5d loss: %.3f' %
                      (i + 1, running_loss / 2000))
                running_loss = 0.0
        idx = np.argsort(np.array(sampler.get_idx()))
        culmulative_orderings = np.array(orderings)[idx]
        return culmulative_orderings

    my_sampler = MySampler(training_set)
    trainloader = torch.utils.data.DataLoader(
        training_set, batch_size=4, shuffle=False, sampler=my_sampler, num_workers=4)

    print("epoch #1")
    real_time_curr = \
        real_time_curriculum(my_sampler, trainloader, model, loss_fn, optimizer)
    epoch = 1
    num_samples = real_time_curr.shape[0]

    for deviation in deviations:
        epoch += 1
        print("epoch #%d" % epoch)
        weights = normal_weights(real_time_curr, np.mean(real_time_curr) + deviation * np.std(real_time_curr))
        weight_denom = np.sum(weights)
        weight_denom = weight_denom if weight_denom != 0 else (1/1e30)
        weights = weights / weight_denom
        sampler = MyWeightedSampler(weights, num_samples, replacement=True)
        real_time_curriculum_loader = \
            torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=False, sampler=sampler, num_workers=4)
        real_time_curr = \
            real_time_curriculum(sampler, real_time_curriculum_loader, model, loss_fn, optimizer)


