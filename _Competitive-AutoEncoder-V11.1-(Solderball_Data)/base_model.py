# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 17:33:46 2021

@author: Chris-Lai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
''' Parameters (Change Anything Here!) '''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


''' CONV-WTA CRITERIA
# - zero padded, so that each feature map has the same size as the input
# - hidden representation is mapped linearly to the output using a deconvolution operation
# - Parameters are optimized to reduce the mean squared error MSE
# - Conv layer is 5 x5, DECONVOLUTION layer is using filters of 11x 11
### In this implementation, I will not use deconvolution, but transpose convolution to ease process
'''
class Competitive_Autoencoder(nn.Module):

    ''' Constructor __init__()
    Params:
        num_features - the number of features you wish to learn in this base/ stack layer
        prev_models - order as follows: [base_Model, stack_1, stack_2......., stack_n]
        prev_num_features - the number of previous layers if this is a Stacked Layer
    '''
    def __init__(self, num_features, k_rate, prev_models = []):
        super().__init__()
        self.k = k_rate
        self.num_features = num_features
        self.prev_models = prev_models
        
        # first get the very last layer of the previous models' # of output features
        self.num_prev_models = len(prev_models)
        
        # assign a single layer or multiple layers as the model depending
        if self.num_prev_models != 0:
            prev_num_features = prev_models[-1].get_numfeatures()
            
            self.encoder_in_use = nn.Sequential(
                nn.Conv2d(prev_num_features, num_features, 5, stride=1, padding = 2),     
                nn.BatchNorm2d(num_features),
            )
            
        else:
            self.encoder_in_use = nn.Sequential(
                nn.Conv2d(3, num_features, 5, stride=1, padding = 2),     
                nn.ReLU(),
                nn.BatchNorm2d(num_features),
                nn.Conv2d(num_features, num_features, 5, stride=1, padding = 2),
                nn.BatchNorm2d(num_features),
            )   
        # Decoder
        self.decoder = nn.ConvTranspose2d(in_channels=num_features, out_channels=3, kernel_size=11, stride =1, padding = 5) # padding will decrease output size # size:N, 28, 28


    ''' TRAINING Feed Forward Function
    - Checks a list of prev_models
    - if there are prev_models, order as follows: [base_Model, stack_1, stack_2......., stack_n]
    - forward() will pass input:x through the ENCODERS of prev_models starting from base_Model
    '''
    # is previous models is a list with the base_model as the first list
    def forward(self, x):
        if (self.num_prev_models != 0):
            # Forward through all of the previous Encoders
            for model in self.prev_models:
                x = model.bottleneck(x)

        encoded = self.encoder_in_use(x)
        winner = self.spatial_sparsity_(encoded)
        self.lifetime_sparsity_(encoded, winner)
        decoded = self.decoder(encoded)
        return decoded
    
    ''' bottleneck()
    - Feeds Forward Input:x up until the bottleneck layer in all the previous models and this model
    - Applies Relu activation because spatial & lifetime is lifted (based on AliReza Paper)
    '''
    def bottleneck(self, x):
        if self.num_prev_models != 0:
            for model in self.prev_models:
                x = model.bottleneck(x)

        return F.relu(self.encoder_in_use(x))
    
    ''' get_numfeatures()
    - returns the number of features used in this layer/base_model
    '''
    def get_numfeatures(self):
        return self.num_features
    
    ''' set_lifetime()
    - returns the number of features used in this layer/base_model
    '''
    def set_lifetime(self, k):
        self.k = k
    
    ''' spatial_sparsity_()
    - Modifies feature map tensors to only retain one, max-valued pixel for every feature, every batch. The rest is set to ZERO
    - With torch.no_grad() temporarily sets all of the requires_grad flags to false (probably not needed)
    '''
    def spatial_sparsity_(self, hiddenMaps):
        with torch.no_grad():
            shape = hiddenMaps.shape  #torch.Size([batch_size, feature_num, 26, 26])
            n_batches = shape[0]
            n_features = shape[1]
            size = shape[2]
            
            # Step 1: flatten it out, find max_vals
            flatten = hiddenMaps.view(n_batches, n_features, -1)
            maxval, _ = torch.max(flatten, 2) # max_val return size[n_batches, n_features]
            
            # Step 2: creating "drop" Array to be multiplied into featureMaps, dropping loser values
            maxval_p = torch.reshape(maxval, (n_batches, n_features, 1, 1))
            drop = torch.where(hiddenMaps < maxval_p, 
                               torch.zeros((n_batches, n_features, size, size)).to(device), 
                               torch.ones((n_batches,n_features, size, size)).to(device))
        
        # To retain the graph, use .data to only modify the data of the tensor
        hiddenMaps.data = hiddenMaps.data*drop.data
        return maxval
        
    ''' spatial_sparsity_()
    - Pick the top-k percent "winner batches" for every feature. The rest of the batches will be zeroed out.
    - With torch.no_grad() temporarily sets all of the requires_grad flags to false (probably not needed)
    '''
    def lifetime_sparsity_(self, hiddenMaps, maxval):
        with torch.no_grad():
            shape = hiddenMaps.shape  #torch.Size([batch_size, feature_num, 26, 26])
            n_batches = shape[0]
            n_features = shape[1]
            
            top_k, _ = torch.topk(maxval, self.k, 0) 

            # Step 2: creating "drop" Array to be multiplied into featureMaps, dropping loser values
            drop = torch.where(maxval < top_k[self.k-1:self.k, :],  
                               torch.zeros((n_batches, n_features)).to(device), 
                               torch.ones((n_batches, n_features)).to(device))

        # To retain the graph, use .data to only modify the data of the tensor
        hiddenMaps.data = hiddenMaps.data * drop.reshape(n_batches, n_features, 1, 1).data