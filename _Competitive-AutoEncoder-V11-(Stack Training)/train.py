# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 17:41:45 2021
@author: Chris-Lai
"""

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
# from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from IPython.display import clear_output
import matplotlib.pyplot as plt
# import torch.nn.functional as F
# import numpy as np
# import math
# import random

from model import Competitive_Autoencoder
import os
import sys


''' Device, Path config'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.append('E:/Chris/Competitive-Autoencoder/')
os.chdir('E:/Chris/Competitive-Autoencoder/')
print("Working on path: ", os.getcwd())

''' Parameters (CHange Anything Here!) '''
transform = transforms.ToTensor()
batch_size = 150

''' Save-Load functions '''
def save_model_optimizer(model, optimizer = None, filename = "CompAutoModel"):
    pthSaves = "_Competitive-AutoEncoder-V11-(Stack Training)/pthSaves/"
    # creating save folder if not already there
    if not os.path.isdir(pthSaves): 
        os.makedirs(pthSaves)
    
    if optimizer == None:
        torch.save({
            'model_state': model.state_dict()
        }, pthSaves+filename+".pth")
    else:
        torch.save({
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict()
        }, pthSaves+filename+".pth")
    
def load_model(filename):
    path = "_Competitive-AutoEncoder-V11-(Stack Training)/pthSaves/"
    checkpoint = torch.load(path+filename+".pth", map_location = 'cpu')
    model_obj = Competitive_Autoencoder()
    model_obj.load_state_dict(checkpoint['model_state'])
    model_obj.eval()
        
    if len(checkpoint.keys()) == 1:
        return model_obj
    else:
        optim_obj = torch.optim.Adam(model_obj.parameters(), lr = 0)
        optim_obj.load_state_dict(checkpoint['optim_state'])
        return model_obj, optim_obj

''' Visualizing functions '''
def train_info_print(loss, loss_mssg, epoch_x, loss_y, epoch):
    # Print information:
    clear_output()
    loss_str = "{:.4f}".format(loss)
    loss_mssg.append('Epoch:' + str(epoch + 1) + ', Loss:' + loss_str)
    print(*loss_mssg, sep = "\n")

    # Graph out the loss:
    loss_y.append(loss.cpu().detach().numpy())
    epoch_x.append(epoch)
    if epoch != 0:
        plt.plot(epoch_x,loss_y)
        plt.show()

def feed_forard_visualize(outputs):
    ##### Plotting the images for convolutional Autoencoder
    
    for k in range(0, len(outputs), 1):
        plt.figure(figsize = (9,2))
        plt.gray()
        
        #because it is a Tensor, so we want to detach and then convert into a numpy array
        imgs = outputs[k][1].cpu().detach().numpy()
        recon = outputs[k][2].cpu().detach().numpy()
        print(imgs.shape)
        for i, item in enumerate(imgs):
            if i >= 9:
                break
            plt.subplot(2, 9, i + 1)
            plt.imshow(item[0])
            
        for i, item in enumerate(recon):
            if i >= 9 :
                break
            plt.subplot(2, 9, 9 + i + 1)
            plt.imshow(item[0])

def deconv_filter_plot(model_load):
    
    model_case = model_load
    
    children = list(model_case.children())
    count = 0
    while(not isinstance(children[count], nn.ConvTranspose2d)):
        count +=1
        
    # printing the layer type
    print(children[count])
    kernels = children[count].weight.detach().clone().cpu()
    print(kernels.shape)
    # normalize to (0,1) range so that matplotlib
    # can plot them
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    print(kernels.shape)
    # kernels is a (Tensor or list) – 4D mini-batch Tensor of shape (B x C x H x W) or a list of images all of the same size
    filter_img = torchvision.utils.make_grid(kernels, nrow = 9)
    # change ordering since matplotlib requires images to 
    # be (H, W, C)
    print(filter_img.shape)
    plt.imshow(filter_img.permute(1, 2, 0))
        
#%%
''' Code Starts Here '''
# Declaring model and stuff
model = Competitive_Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

# Data MNIST
mnist_data = datasets.MNIST(root='./data', train = True, download = False, transform = transform)
data_loader = torch.utils.data.DataLoader(dataset= mnist_data, batch_size = batch_size, shuffle = True)

#%%
# Training Loop
''' Things to try:
- Increase epochs: 25 / (Finding: need to implement stop training methods)
- Include Lifetime Sparsity
- Tweak Batch Sizes (Done 65, 100)
- Stack Training
- batch normalization (THANK GOD)
- os how to not create files if it already exists
- plot loss / (DONE)

WHY the training doesn't converge:
This outcome can happen because there aren’t enough nodes to transform the input data into accurate outputs
or because the architecture has to change drastically in order to better model the data
check:
    - if spatial sparsity really have only 1 value in each channel that is not zero
    - gradient calculations: are they correct? if average over zero values after backward and step
'''

num_epochs = 12
outputs = []
epoch_x = []
loss_y  = []
loss_mssg = []

for epoch in range(num_epochs):
    for (img, _) in data_loader:
        #Reconstructed img, Likely going to be 
        img = img.to(device)
        
        #Forward Pass
        recon  = model(img)
        loss = criterion(recon, img)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Take first 9 examples to be plotted for each # epochs
    if((epoch+1)%2 == 0):
        outputs.append((epoch, img[0:9, :, :, :], recon[0:9, :, :, :], ))
    
    # Printing Training info
    train_info_print(loss, loss_mssg, epoch_x, loss_y, epoch)
        
    # Save the model of the current Epoch (starting point for early ending)
    save_model_optimizer(model, optimizer, "81featuresLifetime5_ep" + str(epoch))

#%%
# Loading the best trained model
which = 10
loadfile = "81featuresLifetime5_ep" + str(which)
model_load, optim_load = load_model(loadfile)

#%%
# Plotting the images for convolutional Autoencoder
feed_forard_visualize(outputs)

# Deconvolution layer filter plotting
deconv_filter_plot(model_load)