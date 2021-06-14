# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 18:19:51 2021
@author: Wei Xun Lai
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from IPython.display import clear_output
import matplotlib.pyplot as plt
import os
import sys
import math
import numpy as np
version = "_Competitive-AutoEncoder-V11.1-(Solderball_Data)"
working_path = "E:/Chris/Competitive-Autoencoder/"
sys.path.append(working_path + version + '/')
from base_model import Competitive_Autoencoder

''' Parameters (CHange Anything Here!) '''
transform = transforms.ToTensor()
batch_size = 50
interface = "spyder"

''' Device, Path config'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.append(working_path)
os.chdir(working_path)
print("Working on path: ", os.getcwd())

''' Save-Load functions '''
def save_model_optimizer(model, save_path, optimizer = None, filename = "default"):
    # creating save folder if not already there
    if not os.path.isdir(save_path): 
        os.makedirs(save_path)
    
    if optimizer == None:
        torch.save({
            'model_state': model.state_dict()
        }, save_path+filename+".pth")
    else:
        torch.save({
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict()
        }, save_path+filename+".pth")
    
def load_model(filename, load_path):
    checkpoint = torch.load(load_path+filename+".pth", map_location = 'cpu')
    model_obj = Competitive_Autoencoder()
    model_obj.load_state_dict(checkpoint['model_state'])
    model_obj.eval()
        
    if len(checkpoint.keys()) == 1:
        return model_obj
    else:
        optim_obj = torch.optim.Adam(model_obj.parameters(), lr = 0)
        optim_obj.load_state_dict(checkpoint['optim_state'])
        return model_obj, optim_obj

def path_creator(starting_num_features, feature_increasing_constant, k_percent, batch_size):
    save_folder = "StackTraining_i" + str(starting_num_features) + "_mul" + str(feature_increasing_constant) + "_L" + str(k_percent) + "_bs" + str(batch_size)
    path = version + "/Results/" + save_folder + "/" # Save and load path
    
    print("Training for: " + save_folder)
    
    if os.path.isdir(path): 
        print("WARNING: " + save_folder + " folder already exists... do you wish to overwrite inside files?\nPress \"e\" to STOP, \"y\" to proceed. Write () reason to create new folder")
        user = input("")
    else:
        return path
    
    if user == "y":
        return path
    elif user == "e":
        sys.exit()
    else:
        save_folder = save_folder + "(" + user +")"
        return version + "/Results/" + save_folder + "/" # Save and load path

''' Visualizing functions '''
def train_info_print(loss, loss_mssg, epoch_x, loss_y, epoch):
    # Print information:
    clear_output()
    loss_str = "{:.4f}".format(loss)
    loss_mssg.append('Epoch:' + str(epoch + 1) + ', Loss:' + loss_str)
    if(interface == "spyder"):
        print(loss_mssg)
    if(interface == "jupyter"):
        print(*loss_mssg, sep = "\n")

    # Graph out the loss:
    # plt.figure()
    # plt.title("Epoch: " + str(epoch)+ ", Loss: " + str(loss))
    # plt.plot(epoch_x,loss_y)
    # plt.show()

def feed_forard_visualize(outputs):
    ##### Plotting the images for convolutional Autoencoder
    
    for k in range(0, len(outputs), 1):
        plt.figure(figsize = (9,2))
        plt.title("Epoch: " + str(outputs[k][0])+ ", Loss: " + str(outputs[k][3]))
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
            
def deconv_filter_plot(model_load, epoch, loss):
    plt.figure()
    plt.title("Epoch: " + str(epoch)+ ", Loss: " + str("{:.4f}".format(loss)))
    model_case = model_load
    
    children = list(model_case.children())
    count = 0
    while(not isinstance(children[count], nn.ConvTranspose2d)):
        count +=1
        
    # printing the layer type
    # print(children[count])
    kernels = children[count].weight.detach().clone().cpu()
    # print(kernels.shape)
    # normalize to (0,1) range so that matplotlib
    # can plot them
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    # print(kernels.shape)
    # kernels is a (Tensor or list) – 4D mini-batch Tensor of shape (B x C x H x W) or a list of images all of the same size
    filter_img = torchvision.utils.make_grid(kernels, nrow = 8)
    # change ordering since matplotlib requires images to 
    # be (H, W, C)
    # print(filter_img.shape)
    plt.imshow(filter_img.permute(1, 2, 0))
    plt.show()

#%%
''' CHANGES IN MODEL TO NOTE:
    - Model now needs a defined num_features, and prev_models = [base_model, stack_1, stack_2,....., stack_n]
'''
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.CenterCrop((120, 120)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std), 
        transforms.RandomCrop((40, 40)),
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

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}

data_loader = dataloaders['train']
# dataiter = iter(data_loader)
# print(next(dataiter))
#%%

'''
Strategy:
    Get base model
'''
k_percent = 5
num_stacks = 1
num_epochs = 40
starting_num_features = 64
feature_increasing_constant = 1
iter_num_features = starting_num_features
prev_models = []
k = math.floor(batch_size*k_percent*0.01)

path = path_creator(starting_num_features, feature_increasing_constant, k_percent, batch_size)

for stack in range(num_stacks):
    
    # Resetting phase to reset everything
    if stack == 0:
        model = Competitive_Autoencoder(starting_num_features, k).to(device)
    else:
        iter_num_features = int(iter_num_features * feature_increasing_constant)
        model = Competitive_Autoencoder(iter_num_features, k, prev_models).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    save_path = path + "stack" + str(stack+1) + "/"
    epoch_x = []
    loss_y  = []
    
    for epoch in range(0, num_epochs):
        
        outputs = []
        loss_mssg = []
        
        print("Tranining stack: " + str(stack+1)+", epoch: "+str(epoch+1))
    
        # All layers are only trained for 1 epoch for now
        for i, (img, _) in enumerate(data_loader):
            #Reconstructed img, Likely going to be 
            img = img.to(device)
            
            #Forward Pass
            recon  = model(img)
            loss = criterion(recon, img)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_x.append(i)
            loss_y.append(loss.cpu().detach().numpy()) 
            
            # Take first 9 examples to be plotted for each # epochs
            # if((i+1)%(math.floor(len(data_loader)/2)) == 0):
                # outputs.append((i, img[0:4, :, :, :], recon[0:4, :, :, :], loss))
                
        # Printing Training info
        train_info_print(loss, loss_mssg, epoch_x, loss_y, i)
        
        # Save the model of the current Epoch (starting point for early ending)
        save_model_optimizer(model, save_path, optimizer, str(iter_num_features) + "ep" + str(i))
        
        # Plotting the images for convolutional Autoencoder
        # feed_forard_visualize(outputs)
    
        # Deconvolution layer filter plotting
        deconv_filter_plot(model, epoch, loss)
    
    # freeze all the layers before appending the model into the prev_arrays
    for para in model.parameters():
        para.requires_grad = False
    prev_models.append(model)

