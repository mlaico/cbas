import numpy as np
from PIL import Image


class FeatureExtractor(object):

    def __init__(self, model=None, embed_layer=None, embed_size=256, transform=None):
        self.embed_size = embed_size
        self.transforms = transform

        if model is None:
            self.model = models.alexnet(pretrained=True)
            self.embed_layer = self.model.features
        else:
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
            print("Warning: converting image to RGB before embedding w/ CNN...")
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
            if len(o.size()) > 2:
                o = o.view(o.size(0), -1)
            embedding.copy_(o.data)

        h = self.embed_layer.register_forward_hook(copy_embedding)
        h_x = self.model(image)
        h.remove()
        return embedding.numpy()

#     def batch_embed(self, imgs):
#         """
#         project a PIL image into embedded feature space, and return that vector as an np array
#         """
# #         print(imgs.shape)
#         a = self.transforms(imgs)
#         images = Variable(a)
# #         image = image.unsqueeze(0)
#         if self.cuda: images.cuda()
#         #print(image.size())

#         embedding = torch.zeros([a.size()[0], self.embed_size])
#         print("sizes: ", a.size(), images.size(), embedding.size())
#         def copy_embedding(m, i, o):
#             if len(o.size()) > 2:
#                 o = o.view(o.size(0), -1)
#             embedding.copy_(o.data)

#         h = self.embed_layer.register_forward_hook(copy_embedding)
#         h_x = self.model(images)
#         h.remove()
#         return embedding.numpy()
