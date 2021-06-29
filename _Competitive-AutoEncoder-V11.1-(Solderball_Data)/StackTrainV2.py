# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 17:02:29 2021
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
from stacked_model import Competitive_Autoencoder
from Competitive_Segmentator import Competitive_Segmentator
from PIL import Image, ImageOps

#%% Parameters and Device Config
''' Parameters (CHange Anything Here!) '''
batch_size = 100
interface = "spyder"

''' Device, Path config'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sys.path.append(working_path)
os.chdir(working_path)
print("Working on path: ", os.getcwd())

#%% Functions

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
    
def load_model(model_obj, path):
    checkpoint = torch.load(path, map_location = 'cpu')
    model_obj.load_state_dict(checkpoint['model_state'])
    model_obj.eval()
        
    if len(checkpoint.keys()) == 1:
        return model_obj
    else:
        optim_obj = torch.optim.Adam(model_obj.parameters(), lr = 0)
        optim_obj.load_state_dict(checkpoint['optim_state'])

def load_best(save_path, stack, iter_num_features, k, prev_models):
    
    not_loaded = True
    
    while(not_loaded):
        print("Which model epoch would you like to retain from stack"+str(stack+1)+"?")
        user = input("")
        
        filename = str(iter_num_features)+"ep"+user+".pth"
        path = os.path.join(save_path, filename)
        if os.path.exists(path):
            model_obj = Competitive_Autoencoder(iter_num_features, k, prev_models).to(device)
            load_model(model_obj, path)
            not_loaded = False
        else:
            print("Filename Does Not Exist!\n")
            
    print("Model for epoch "+user+" Successfully loaded!")
    return model_obj

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
        # print(imgs.shape)
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
            
def deconv_filter_plot(model_load, num_features, epoch = -1, loss = -1):
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
    # normalize to (0,1) range so that matplotlib can plot them
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    # kernels is a (Tensor or list) – 4D mini-batch Tensor of shape (B x C x H x W) or a list of images all of the same size
    filter_img = torchvision.utils.make_grid(kernels, nrow = int(math.sqrt(num_features)))
    # change ordering since matplotlib requires images to be (H, W, C)
    # print(filter_img.shape)
    plt.imshow(filter_img.permute(1, 2, 0))
    plt.show()

def new_deconv_filter_plot(model_load, num_features, epoch = -1, loss = -1):

    model_case = model_load
    
    children = list(model_case.children())
    count = 0
    while(not isinstance(children[count], nn.ConvTranspose2d)):
        count +=1
        
    # printing the layer type
    # print(children[count])
    kernels = children[count].weight.detach().clone().cpu()
    # normalize to (0,1) range so that matplotlib can plot them
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    kernels = kernels.permute(0, 2, 3, 1).numpy() * 255
    kernels = kernels.astype(np.uint8)
    
    # print(kernels)
    # kernels = (255.0 / kernels.max() * (kernels - kernels.min())).permute(0, 2, 3, 1).numpy().astype(np.uint8)
    
    # kernels = np.transpose(kernels, (0, 2, 3, 1))
    return concat_images(kernels, (kernels.shape[1], kernels.shape[1]), 1)
    
def dataloader_example(image_datasets, dataloaders, which):
    # dataset_sizes = {x: len(image_datasets[x]) for x in ['train','truth', 'val']}
    class_names = image_datasets['testdata'].classes
    
    def imshow(inp, title):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        # plt.title(title)
        plt.show()
    
    
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders[which]))
    
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    
    imshow(out, title=[class_names[x] for x in classes])

def concat_images(features, size, border_width = 2, grey = False):
    
    assert type(features).__module__ == np.__name__ , "Features: should be a numpy array"  # assert features to be numpy array
    assert border_width >= 0,"border_width: should be a Natural Number"
    
    # auto assert shape to always be a square fit no matter how many features
    temp = math.sqrt(len(features))
    temp1 = int(math.floor(temp))
    shape = (temp1+1 if temp1<temp else temp1, temp1)
    
    # Open images into an array called "images"
    size_1 = (size[0] + border_width*2, size[1] + border_width*2)
    width, height = size_1
    # print(size_1)
    images = [Image.fromarray(feature).convert("RGB") for feature in features]
    # images = map(Image.open, image_paths)
    color = 255 if grey else (165,42,42)
    ## Adding border to every image 
    images = [ImageOps.expand(image,border=border_width,fill=color)
                       for image in images]
    
    # Create canvas for the final image with total size
    shape = shape if shape else (1, len(images)) # if shape not given then all will be organized in a row
    image_size = (width * shape[1], height * shape[0])
    image = Image.new('L', image_size) if grey else Image.new('RGB', image_size)
    
    # Paste images into final image
    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            # return if reach the desired image value
            if idx >= len(features):
                return ImageOps.expand(image,border=border_width + 1,fill=color)
            image.paste(images[idx], offset)
    
    return ImageOps.expand(image,border=border_width + 1,fill=color)

#%% Data, Datasets, Dataloaders
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
        # transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(mean, std), 
        transforms.RandomCrop((40, 40)),
    ]),
    'testdata': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.CenterCrop((40, 40)),
        transforms.ToTensor(),
        # transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(mean, std), 
        transforms.RandomCrop((40, 40)),
    ]),
    'testlabels': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.CenterCrop((40, 40)),
        transforms.ToTensor(),
        # transforms.Grayscale(num_output_channels=1),
    ]),
    'truth': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean, std), 
    ]),
    'truth_gauss': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std), 
    ]),
    'val': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.CenterCrop((120, 120)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'train_gauss': transforms.Compose([
        transforms.CenterCrop((120, 120)),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std), 
        transforms.RandomCrop((40, 40)),
    ]),
}

data_dir = 'data/InariPads/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'train_gauss', 'truth', 'truth_gauss', 'testdata', 'testlabels']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'train_gauss', 'truth', 'truth_gauss', 'testdata', 'testlabels']}

datatesters= {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=False, num_workers=0)
              for x in ['testdata', 'testlabels']}
# data_booster = dataloaders['truth']
# data_booster1 = dataloaders['truth1']
# data_loader = dataloaders['train_gauss']

dataloader_example(image_datasets, datatesters, 'testdata')
dataloader_example(image_datasets, datatesters, 'testlabels')
# dataloader_example(image_datasets, dataloaders, 'truth')
# dataloader_example(image_datasets, dataloaders, 'train_edge')

#%% Training Parameters

data_loader = dataloaders['train_gauss']
k_percent = 5
num_stacks = 1
num_start_epochs = 30
num_incr_epochs = 10
starting_num_features = 128
feature_increasing_constant = 1
iter_num_epochs   = num_start_epochs
iter_num_features = starting_num_features
prev_models = []
k = max(math.floor(batch_size*k_percent*0.01), 1)
criterion = nn.MSELoss()


#%% Creating and Training the Model

'''
Strategy:
    Start with 1st stack, then moving on 
'''

# Ensure that you want to overwrite the path created
path = path_creator(starting_num_features, feature_increasing_constant, k_percent, batch_size)

# Hirarchial training loop
for stack in range(num_stacks):
    
    # Creating new model.
    if stack == 0:
        model = Competitive_Autoencoder(starting_num_features, k).to(device)
    else:
        iter_num_epochs = iter_num_epochs + num_incr_epochs
        iter_num_features = int(iter_num_features * feature_increasing_constant)
        model = Competitive_Autoencoder(iter_num_features, k, prev_models).to(device)
    
    # Resetting variables before training next stacked model
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    save_path = path + "stack" + str(stack+1) + "/"
    epoch_x = []
    loss_y  = []
    
    # Training Loop
    for epoch in range(0, iter_num_epochs):
        
        outputs = []
        loss_mssg = []
        
        print("Tranining stack: " + str(stack+1)+", epoch: "+str(epoch+1))
    
        '''Usual training'''
        # All layers are only trained for 1 epoch for now
        for i, (img, _) in enumerate(data_loader):
            #Reconstructed img, Likely going to be 
            img = img.to(device)
            
            # Forward Pass. Note: input_img should become smaller as stack deepens
            recon, input_img = model(img)
            loss = criterion(recon, input_img)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update visualizing purpose parameters
            epoch_x.append(i)
            loss_y.append(loss.cpu().detach().numpy()) 
            
            # Take first 6 examples to be plotted for each # epochs
            if(((i+1)%(math.floor(len(data_loader)/2)) == 0) and stack == 0):
                outputs.append((i, img[0:6, :, :, :], recon[0:6, :, :, :], loss))
                
        # Printing Training info
        train_info_print(loss, loss_mssg, epoch_x, loss_y, i)
        
        # Save the model of the current Epoch (starting point for early ending)
        save_model_optimizer(model, save_path, optimizer, str(iter_num_features) + "ep" + str(epoch))
        
        # Only the first layer is visualizable
        if(stack == 0):
            # Plotting the images for convolutional Autoencoder
            feed_forard_visualize(outputs)
        
            # Deconvolution layer filter plotting
            deconv_filter_plot(model, iter_num_features, epoch, loss)
        
        '''
        ## Truth training
        # truth_epochs = 4000
        # # model.set_lifetime(5)
        # for i in range(0, truth_epochs):

        #     # All layers are only trained for 1 epoch for now
        #     for (img, _) in data_booster:
        #         #Reconstructed img, Likely going to be 
        #         img = img.to(device)
                
        #         #Forward Pass
        #         recon  = model(img)
        #         loss = criterion(recon, img)
                
        #         # Backward and optimize
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        
        #     if(i%200 == 0):
        #         # Deconvolution layer filter plotting
        #         deconv_filter_plot(model, iter_num_features, i + 0.1, loss)
                             
        #         # Plotting the images for convolutional Autoencoder
        #         outputs = [(i, img[0:5, :, :, :], recon[0:5, :, :, :], loss)]
        #         feed_forard_visualize(outputs)

        #         # # Save the model of the current Epoch (starting point for early ending)
        #         save_model_optimizer(model, save_path, optimizer, str(iter_num_features) + "ep" + str(i))
                
        # setting back thelifetime value
        # model.set_lifetime(k)
        '''
    
    # Ask user for the best model for this stack and load it
    model = load_best(save_path, stack, iter_num_features, k, prev_models)
    
    # freeze all the layers before appending the model into the prev_arrays
    for para in model.parameters():
        para.requires_grad = False
    prev_models.append(model)

#%% Adding Segmentator Component
data_loader = datatesters['testdata']
data_labels = datatesters['testlabel']
learning_rate = 0.001
model1 = Competitive_Autoencoder(prev_models).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
#%% Segmentator Training Loop
for epoch in range(0, iter_num_epochs):
    
    outputs = []
    loss_mssg = []
    
    print("Tranining stack: " + str(stack+1)+", epoch: "+str(epoch+1))

    '''Usual training'''
    # All layers are only trained for 1 epoch for now
    for i, (img, label) in enumerate(data_loader):
        #Reconstructed img, Likely going to be 
        img = img.to(device)
        
        # Forward Pass. Note: input_img should become smaller as stack deepens
        recon = model(img)
        loss = criterion(recon, label)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update visualizing purpose parameters
        epoch_x.append(i)
        loss_y.append(loss.cpu().detach().numpy()) 
        
        # Take first 6 examples to be plotted for each # epochs
        if(((i+1)%(math.floor(len(data_loader)-1/2)) == 0) and stack == 0):
            outputs.append((i, img[0:6, :, :, :], recon[0:6, :, :, :], loss))
            
    # Printing Training info
    train_info_print(loss, loss_mssg, epoch_x, loss_y, i)
    
    # Save the model of the current Epoch (starting point for early ending)
    save_model_optimizer(model, save_path, optimizer, str(iter_num_features) + "ep" + str(epoch))
    
    # Only the first layer is visualizable
    if(stack == 0):
        # Plotting the images for convolutional Autoencoder
        feed_forard_visualize(outputs)
    
        # Deconvolution layer filter plotting
        deconv_filter_plot(model, iter_num_features, epoch, loss)
