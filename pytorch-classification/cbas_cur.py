'''
Training script for CBAS-34
'''
from __future__ import print_function

import sys
sys.path.append('..')
sys.path.append('../../PythonAPI')

import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models

from cbas import CBAS

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CBAS-34 Training')
# Datasets
parser.add_argument('-d', '--dataset', default='cbas34', type=str)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--curr', default='none', type=str)
# Optimization options
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=8, help='Model cardinality (group).')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
parser.add_argument('--growthRate', type=int, default=12, help='Growth rate for DenseNet.')
parser.add_argument('--compressionRate', type=int, default=2, help='Compression Rate (theta) for DenseNet.')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Validate dataset
assert args.dataset == 'cbas34' or args.dataset == 'cbasLS', 'Dataset can only be cbas34 or cbasLS.'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)


    # Data
    print('==> Preparing dataset %s' % args.dataset)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4587, 0.4291, 0.3874), (0.1974, 0.1937, 0.1941)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4587, 0.4291, 0.3874), (0.1974, 0.1937, 0.1941)),
    ])
    if args.dataset == 'cbas34':
        dataloader = datasets.ImageFolder
        num_classes = 34
    else:
        dataloader = datasets.ImageFolder
        num_classes = 34

    # < Curriculum code start >

    cbas_api = CBAS('../cbas80.json') # cbas splits coco train2017 into train and val

    imageSets = {}
    imageSets['train'] = datasets.ImageFolder(root='../../images/cbas34_train', transform=transform_train)
    imageSets['val'] = datasets.ImageFolder(root='../../images/cbas34_val', transform=transform_test)

    # Compute CBAS curriculum intervals
    SIZE=32
    image_pixels = float(SIZE*SIZE)
    pixel_counts = [0.0,16.0,64.0,128.0,256.0,1024.0]
    BINS = []
    if args.curr == "random":
        print('Creating a randomized curriculum')
        imsetIds = {}
        interval_size = {}
        interval_size['train'] = 10200
        interval_size['val'] = 1088

        for dataType in ['train','val']:
                imsetIds[dataType] = []
                for i,_ in enumerate(imageSets[dataType].imgs):
                    imsetIds[dataType].append(i)
                random.shuffle(imsetIds[dataType])

        all_loaders = []
        for b in range(1,6):
            imageLoaders = {}

            for dataType in ['train','val']:
                samplerIndices = imsetIds[dataType][0:b*interval_size[dataType]]
                curSplit = torch.utils.data.sampler.SubsetRandomSampler(samplerIndices)
                imageLoaders[dataType] = torch.utils.data.DataLoader(imageSets[dataType],
                    batch_size=args.train_batch, shuffle=False, sampler=curSplit, num_workers=args.workers)
            all_loaders.append(imageLoaders)
    else:
        if args.curr == 'start-big':
            for i in range(len(pixel_counts)-2, -1, -1): # reverse order
                BINS.append( [ (pixel_counts[i]/image_pixels), (pixel_counts[-1]/image_pixels) ] )
        elif args.curr == 'start-small':
            for i in range(0,len(pixel_counts)-1):
                BINS.append( [ (pixel_counts[0]/image_pixels), (pixel_counts[i+1]/image_pixels) ] ) # cumulative
        elif args.curr == 'middle-out':
            BINS.append([ (pixel_counts[3]/image_pixels), (pixel_counts[4]/image_pixels) ])
            BINS.append([ (pixel_counts[2]/image_pixels), (pixel_counts[4]/image_pixels) ])
            BINS.append([ (pixel_counts[2]/image_pixels), (pixel_counts[5]/image_pixels) ])
            BINS.append([ (pixel_counts[1]/image_pixels), (pixel_counts[5]/image_pixels) ])
        else:
            BINS = [[0.0,1.0]]

        if not args.curr == 'none':
            print('Creating a curriculum for the following size bounds: {}'.format(BINS))

        all_loaders = []
        for b in BINS:
            imageLoaders = {}

            for dataType in ['train','val']:
                idToIdx = {}
                datasetIds = []
                for i,img in enumerate(imageSets[dataType].imgs):
                    img_id_str = img[0].split('/')[-1].split('.')[0]
                    idToIdx[img_id_str] = i
                    datasetIds.append(img_id_str)

            # Get list of ids in size range and intersect with list of ids in dataset
                imgIds_in_size_range = cbas_api.getImgIds(imgIds=cbas_api.getImgIds(), szBounds=b)
                imgIds_szRange = [str(i) for i in imgIds_in_size_range] # change list elements to strings

                imgIds_for_sampling = list(set(datasetIds) & set(imgIds_szRange))
                samplerIndices = [idToIdx[i] for i in imgIds_for_sampling]

                curSplit = torch.utils.data.sampler.SubsetRandomSampler(samplerIndices)
                imageLoaders[dataType] = torch.utils.data.DataLoader(imageSets[dataType],
                    batch_size=args.train_batch, shuffle=False, sampler=curSplit, num_workers=args.workers)
            all_loaders.append(imageLoaders)

    # < End curriculum code >

    # Model
    print("==> creating model '{}'".format(args.arch))
    if args.arch.startswith('resnext'):
        model = models.__dict__[args.arch](
                    cardinality=args.cardinality,
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('densenet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    growthRate=args.growthRate,
                    compressionRate=args.compressionRate,
                    dropRate=args.drop,
                )
    elif args.arch.startswith('wrn'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                    widen_factor=args.widen_factor,
                    dropRate=args.drop,
                )
    elif args.arch.endswith('resnet'):
        model = models.__dict__[args.arch](
                    num_classes=num_classes,
                    depth=args.depth,
                )

    else:
        model = models.__dict__[args.arch](num_classes=num_classes)

    if use_cuda:
        #model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
        cudnn.benchmark = True

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    title = 'cbas-34-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])


    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # loop over curriculum batch loaders
    for i,loader in enumerate(all_loaders):

        if not args.curr == 'none':
            if args.curr == 'random':
                print('\nloading random curriculum')
            else:
                print('\nLoading curriculum {}: {}'.format(i+1,BINS[i]))

        # Train and val
        for epoch in range(start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)

            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

            print(len(loader['train']))

            train_loss, train_acc = train(loader['train'], model, criterion, optimizer, epoch, use_cuda)
            test_loss, test_acc = test(loader['val'], model, criterion, epoch, use_cuda)

            # append logger file
            logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

            # save model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)

def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()
