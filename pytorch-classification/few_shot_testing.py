import sys
sys.path.append('../../PythonAPI')
sys.path.append('../../PythonAPI/pycocotools')
sys.path.append('../')
sys.path.append("./pytorch-classification")
sys.path.append("./pytorch-classification/models")
sys.path.append("./pytorch-classification/models/cifar/")
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
# from cbas import CBAS
from pycocotools.coco import COCO
# from pycocotools.cbas import CBAS
import cbas_construction_utils as ccu
import sklearn as sk
import argparse
import os
import cbas
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import models.cifar as cbasmodels
from collections import OrderedDict
import numpy as np
from PIL import Image
# from cbas import FeatureExtractor
import glob
from sklearn.metrics.pairwise import cosine_similarity
import sklearn as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch._utils

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


IMGS_BASE = '../../images/'

class FeatureExtractor(object):

    def __init__(self, model=None, embed_layer=None, embed_size=256, transform=None):
        self.embed_size = embed_size
        self.transforms = transform

        if model is None:
            self.model = models.alexnet(pretrained=True)
            self.embed_layer = self.model.features
        else:
            print("embed_size: ", embed_size)
            if embed_layer is None:
                raise ValueError("Need to specify embed_layer if you pass in a model to FeatureExtractor!")
            self.model = model
            self.embed_layer = embed_layer

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.model.cuda()

        # Set model to eval mode so any train-specific things like dropout, etc. don't run:
        self.model.eval()


    def embed(self, img):
        """
        project a PIL image into embedded feature space, and return that vector as an np array
        """
        # Work arround issue of some of the train images not being RGB, (they won't go through the CNN):
        if img.mode != "RGB":
            # print("Warning: converting image to RGB before embedding w/ CNN...")
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            img = rgbimg

        a = self.transforms(img)
        image = Variable(a)
        image = image.unsqueeze(0)
        if self.cuda: image.cuda()
        # print(image.size())

        embedding = torch.zeros(self.embed_size)

        def copy_embedding(m, i, o):
            # print("Size: ", o.size())
            if len(o.size()) > 2:
                o = o.view(o.size(0), -1)
            embedding.copy_(o.data)

        h = self.embed_layer.register_forward_hook(copy_embedding)
        h_x = self.model(image)
        h.remove()
        return embedding.numpy()


def get_cos_similarities(base_img_path, other_image_paths):
    img = Image.open(base_img_path)
    # Work arround issue of some of the train images not being RGB, (they won't go through the CNN):
    if img.mode != "RGB":
        # print("Warning: converting {} to rgb...".format(base_img_path))
        rgbimg = Image.new("RGB", img.size)
        rgbimg.paste(img)
        img = rgbimg
    base_embedding = embedder.embed(img)
    similarities = np.zeros((len(other_image_paths)))

    for idx, img_path in enumerate(other_image_paths):
        # print("idx: ", idx, img_path)
        embedding = None
        img = Image.open(img_path)
        try:
            embedding = embedder.embed(img)
        except:
            print("error on image #", idx)
        if embedding is not None:
            #             cos_sim = cosine_similarity(base_embedding, embedding)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            if use_cuda:
                cos_sim = cos(torch.Tensor(base_embedding).cuda().unsqueeze(0),
                              torch.Tensor(embedding).cuda().unsqueeze(0))
            else:
                cos_sim = cos(torch.Tensor(base_embedding).unsqueeze(0), torch.Tensor(embedding).unsqueeze(0))
            similarities[idx] = cos_sim.cpu().numpy()
    return similarities


def avg_category_similarities(dataset_path, cat1, cat2, sample_size=50):
    """
    Compute avereage similarities between all imgs in categories cat1 an cat2
    """
    cat1_imgs = glob.glob(dataset_path.format(cat1) + '*.jpg')
    cat2_imgs = glob.glob(dataset_path.format(cat2) + '*.jpg')
    # Sample a limited size to reduce compute time:
    size1, size2 = min(len(cat1_imgs), sample_size), min(len(cat2_imgs), sample_size)
    cat1_imgs = np.random.choice(cat1_imgs, size=size1)
    cat2_imgs = np.random.choice(cat2_imgs, size=size2)
    sims = np.zeros((size1, size2))
    # print("similarities shape: ", sims.shape)
    for i, img1 in enumerate(cat1_imgs):
        sims[i, :] = get_cos_similarities(img1, cat2_imgs).T
        pass
    return sims


def get_ls_holdout_categories():
    return np.sort([c.replace(IMGS_BASE + 'cbasLS_train/', '').replace('/', '')
                    for c in glob.glob(IMGS_BASE + 'cbasLS_train/*/')]
                   )


def get_joint_categories():
    joint_cats = []
    base_cats = glob.glob(IMGS_BASE + 'cbas34_train/*/')
    holdout_cats = glob.glob(IMGS_BASE + 'cbasLS_train/*/')
    joint_cats.extend(base_cats)
    joint_cats.extend(holdout_cats)
    return joint_cats


# For each category in cbasLS, choose N examples from cbasLS train set:
def get_holdout_train_examples(n):
    """
    Gets a list of image paths from cbasLS train set. List contains n
    randomly sampled images from each category in cbasLS
    """
    ls_path = IMGS_BASE + 'cbasLS_train/'
    cats = glob.glob(ls_path + '*/')
    train_example_paths = []
    # print(len(cats), cats)
    for cat_path in cats:
        imgs = glob.glob("{}*.jpg".format(cat_path))
        train_example_paths.extend(np.random.choice(imgs, replace=False, size=n))
    return train_example_paths


def get_embeddings(embedder, img_paths):
    """
    input: is list of image paths.
    output: list of tuples of form: (category, id, path, vec. embedding)
    """
    return [
        (os.path.dirname(p).split('/')[-1]
         , os.path.basename(p).replace('.jpg', '')
         , p
         , embedder.embed(Image.open(p))
         ) for p in img_paths
    ]


# Get embeddings for images in holdout validation set:
def get_ls_valid_embeddings(embedder):
    """
    output: list of tuples of form: (category, id, path, vec. embedding)
    """
    valid_paths = glob.glob(IMGS_BASE + 'cbasLS_val/*/*.jpg')
    print("Found {} validation images in cbasLS_val. Getting embeddings...".format(len(valid_paths)))
    embeddings = get_embeddings(embedder, valid_paths)
    return embeddings


def lowshot_valid_acc(X_valid, y_valid, y_pred, y_pred_probs=None, cats=[]):
    """
    inputs:
        X_valid: list of data features that were predicted from (e.g., list of img vector embeddings)
        y_pred: list of same length as X_valid, containing predicted values
        y_pred_probs: list or vector of probabilities. dimension should be: [len(X_valid), len(cats)]
        cats: list of target categories, AKA list of ALL possible predicted values in y_pred.
                only needed if y_pred_probs is not None. It's used in calculating top-5 accuracy

    Returns validation accuracy. If y_pred_probs is passed in (and not None)
        , also returns top-5 accuracy.
    """
    if not ((len(X_valid) == len(y_valid)) and (len(X_valid) == len(y_pred))):
        raise ValueError(
            "Lengths must be the same for: X_valid:{}, y_valid:{}, y_pred:{}".format(
                len(X_valid), len(y_valid), len(y_pred)
            ))

    total, total_acc, topN_acc = len(X_valid), 0, 0

    if y_pred_probs is not None:
        topN = 5  # N as in "top-5 accuracy" -> N=5
        cats_sorted = np.sort(cats)
    else:
        cats_sorted = None

    for i, x_val in enumerate(X_valid):
        total_acc += 1 if y_valid[i] == y_pred[i] else 0
        if y_pred_probs is not None:
            # indices of topN probabilities:
            topN_indices = np.argsort(y_pred_probs[i])[-topN:]
            # print("all_cats_sort: ", cats_sorted.shape, cats_sorted)
            top_predicted_cats = cats_sorted[topN_indices]
            #             print(y_pred_probs.shape)
            #             print("\ny_pred: {}, topN_pred: {}".format(y_pred[i], top_predicted_cats))
            #             print(y_pred_probs)
            topN_acc += 1 if y_valid[i] in top_predicted_cats else 0

    val_acc = float(total_acc) / float(total)
    if y_pred_probs is not None:
        topN_val_acc = float(topN_acc) / float(total)
    # Turns out sk has acc score function, it returns same value as val_acc:
    # print("sk_acc: ", accuracy_score(y_valid, y_pred, normalize=True))

    return val_acc, topN_val_acc if y_pred_probs is not None else None


def lowshot_fit_and_predict(embedder, N, knn_args, target_set="holdout"):
    """
    Trains knn classifier with N training examples from each category in cbasLS,
    and returns validation accuracy on entire cbasLS_val set
    inputs:
        N: num. training examples per cbasLS_train category
        target_set: (str) indicates which set of categories we train classifier with.
                    Should either be: 'holdout' for holdout set only (harder task),
                    or 'joint' for both holdout and train/base sets.
    """
    if target_set!="holdout":
        raise NotImplementedError("Only support for target_set='holdout' is available rn.")
    train_paths = get_holdout_train_examples(N)
    train_samples = get_embeddings(embedder, train_paths)
    # print(train_samples)
    all_cats = get_ls_holdout_categories()
    #print("all_cats: ", len(all_cats), all_cats)
    #print("Found {} total (base + holdout) cats".format(len(joint_cats)))

    # Fit a knn classifier to those N training examples:
    X_train = [vec for cat,id,p,vec in train_samples]
    y_train = [cat for cat,id,p,vec in train_samples]
    knn = KNeighborsClassifier(**knn_args)
    knn.fit(X_train, y_train)

    ls_valid_set = get_ls_valid_embeddings(embedder)
    #print(ls_valid_set[0])
    X_valid = [vec for cat,id,p,vec in ls_valid_set]
    y_valid = [cat for cat,id,p,vec in ls_valid_set]
    y_pred = knn.predict(X_valid)
    y_pred_probs = knn.predict_proba(X_valid)

    # Val acc:
    return lowshot_valid_acc(X_valid, y_valid, y_pred, y_pred_probs=y_pred_probs, cats=all_cats)


def load_model(path, model=None):
    gpu_id = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    use_cuda = torch.cuda.is_available()
    print("use_cuda: ", use_cuda)
    print("Loading model from: ", path)
    # print("Model: ", model)

    if model is None:
        model = cbasmodels.alexnet(num_classes=34)

    if use_cuda:
        print("Using GPU")
        model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
        cudnn.benchmark = True
        checkpoint = torch.load(path)
        state_dict = checkpoint['state_dict']
    else:
        print("Using CPU (no GPU)")
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['state_dict']
        trained_on_gpu = list(state_dict.keys())[0].startswith('module.')
        print(state_dict.keys())
        if trained_on_gpu:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove module.
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint['state_dict']

    #     print(checkpoint.keys())
    # print("state_dict: ", checkpoint['state_dict'].keys())
    model.load_state_dict(state_dict)
    return model, use_cuda


def get_model_from_type(model_type):
    if model_type=='dilenet':
        return cbasmodels.dilenet()
    elif model_type=='alexnet':
        return cbasmodels.alexnet(num_classes=34)
    else:
        raise ValueError("Unsupported model_type '{}'".format(model_type))


def few_shot_fit_and_eval(
        model_info
):
    model_type, curric_type, model_path = model_info
    print("Running fewshot training and eval on modeltype: ", model_type)

    ## Load Model:
    model, use_cuda = load_model(model_path, model=get_model_from_type(model_type))
    print(use_cuda, model)

    ## Create embedder:
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    if model_type == 'dilenet':
        embed_layer = model.module.features if use_cuda else model.features
        embed_size = 4096
    elif model_type == 'alexnet':
        embed_layer = model.module.features if use_cuda else model.features
        embed_size = 256
    if use_cuda:
        embedder = FeatureExtractor(model=model, transform=transform, embed_layer=embed_layer, embed_size=embed_size)
    else:
        embedder = FeatureExtractor(model=model, transform=transform, embed_layer=embed_layer, embed_size=embed_size)

    ## Now do fit and eval:
    N_vals = [1, 2, 5, 10, 20]
    # Args that are forwarded to sklearn's KNN classifier:
    results = np.zeros((len(N_vals), 2))
    for i,N in enumerate(N_vals):
        # Right now use k=N, but there is no reason to do this. we should experiment with fixed values of k
        # if there is time:
        knn_args = {
            "n_neighbors": 3
        }
        print("Training lowshot predictor for N={}".format(N))
        accs = lowshot_fit_and_predict(embedder, N, knn_args)
        print("Validation - Top-1: {}, Top-5: {}".format(accs[0], accs[1]))
        results[i,0] = accs[0]
        results[i, 1] = accs[1]

    ## Print results in markdown table format:
    print("Final results:")
    print("### Top-1:")
    print("| Curriculum   |	Model     | Conv (eps) | n=1  | 2  | 5  | 	10  | 20 |")
    top1_row = list(map(str, results[:,0]))
    print(
        "| {}         |  {}   |     ??     |  {} | ".format(
            curric_type
            , model_type
            , " | ".join(top1_row)
        )
    )

    print("### Top-5:")
    print("| Curriculum   |	Model     | Conv (eps) | n=1  | 2  | 5  | 	10  | 20 |")
    top5_row = list(map(str, results[:,1]))
    print(
        "| {}         |  {}   |     ??     |  {} | ".format(
            curric_type
            , model_type
            , " | ".join(top5_row)
        )
    )


if __name__ == '__main__':
    models = [
        ## Done:
        # ('dilenet', 'None', '/home/gbiamby/school/coco/cbas/models/dilent_cbas34_no-curr.pth.tar')
        # ('alexnet', 'None', '/home/gbiamby/school/coco/cbas/pytorch-classification/checkpoints/cbas34/alexnet/alexnet_300epochs_no-curr.pth.tar')

        ## Not Done:
        ('dilenet', 'Start Small', '/home/gbiamby/school/coco/cbas/pytorch-classification/checkpoints/cbas34/dilenet/start-small/model_best.pth.tar')
        , ('dilenet', 'Start Big', '/home/gbiamby/school/coco/cbas/pytorch-classification/checkpoints/cbas34/dilenet/start-big/model_best.pth.tar')
        , ('dilenet', 'Middle-out', '/home/gbiamby/school/coco/cbas/pytorch-classification/checkpoints/cbas34/dilenet/middle-out/model_best.pth.tar')
    ]

    for model_info in models:
        few_shot_fit_and_eval(model_info)

