# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 22:15:18 2021

@author: V510
"""

import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import sys

working_path = "E:/Chris/Competitive-Autoencoder/"
sys.path.append(working_path)


#%%
'''
@source https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
'''
from skimage import io, transform, filters

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        # assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # h, w = image.shape[:2]
        # if isinstance(self.output_size, int):
        #     if h > w:
        #         new_h, new_w = self.output_size * h / w, self.output_size
        #     else:
        #         new_h, new_w = self.output_size, self.output_size * w / h
        # else:
        #     new_h, new_w = self.output_size

        # new_h, new_w = int(new_h), int(new_w)

        img = filters.gaussian(image, sigma=self.output_size)

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}
#%%


mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.CenterCrop((120, 120)),
        # transforms.RandomCrop((30, 30)),
        transforms.ToTensor(),
        Rescale(2.375),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.CenterCrop((120, 120)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

data_dir = 'data/InariPads/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(class_names)


def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

#%%
import scipy.ndimage.filters as nd
from PIL import Image
from skimage import io, transform, filters

# Read image
img = Image.open('data/InariPads/train/padclass/4-1-bm6_img.jpg')
  
# Output Images
img.show()
  
# prints format of image
print(img.format)
  
# prints mode of image
print(img.mode)

print(img.shape)
resize = transform.resize(img, (200, 200))
print(resize.shape)
gaussian = nd.gaussian_filter(img, 2.375)
print(gaussian.shape)
plt.imshow(gaussian)
plt.show()