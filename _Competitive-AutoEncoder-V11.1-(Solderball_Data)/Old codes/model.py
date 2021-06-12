# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 17:33:46 2021

@author: Chris-Lai
"""

import torch
import torch.nn as nn

''' Parameters (Change Anything Here!) '''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
k_rate = 0.05

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
        
    '''
    # num_features should increase as layers increase?
    def __init__(self, num_features, prev_num_features = None):
        super().__init__()
            
        #Image size:N, 28, 28
        # Notes:
        #   Final with and without relu almost the same
        #   Base layer would first try to minimize the loss
        #   number of features would be 32, 64, 128
        
        # Encoders
        self.base_encoder = nn.Sequential(
            nn.Conv2d(1, num_features, 5, stride=1, padding = 2),     
            nn.ReLU(),
            nn.BatchNorm2d(num_features),
            nn.Conv2d(num_features, num_features, 5, stride=1, padding = 2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features),
        )
        
        self.stack_encoder = nn.Sequential(
            nn.Conv2d(prev_num_features, num_features, 5, stride=1, padding = 2),     
            nn.ReLU(),
            nn.BatchNorm2d(num_features),
        )
        
        # Decoder
        self.decoder = nn.ConvTranspose2d(in_channels=num_features, out_channels=1, kernel_size=11, stride =1, padding = 5) # padding will decrease output size # size:N, 28, 28
        
        # encoder_in_use attribute -- if no previous_num_feature is given, it is base model
        if prev_num_features == None:
            self.encoder_in_use = self.base_encoder
        else:
            self.encoder_in_use = self.stack_encoder
    
    ''' Feed Forward Function
    - Checks a list of prev_models
    - if there are prev_models, order as follows: [base_Model, stack_1, stack_2......., stack_n]
    - forward() will pass input:x through the ENCODERS of prev_models starting from base_Model
    '''
    # is previous models is a list with the base_model as the first list
    def forward(self, x, prev_models = []):
        if (len(prev_models) == 0):
            # need to forward through all of the previous models
            for model in prev_models:
                x = model.bottleneck(x)
        else:
            encoder_shell = self.encoder_in_use
            
        encoded = encoder_shell(x)
        winner = self.spatial_sparsity_(encoded)
        self.lifetime_sparsity_(encoded, winner, k_rate)
        decoded = self.decoder(encoded)
        return decoded
    
    ''' bottleneck()
    - Feeds Forward Input:x up until the bottleneck layer
    '''
    def bottleneck(self, x):
        return self.encoder_in_use(x)
    
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
    def lifetime_sparsity_(self, hiddenMaps, maxval, k_percent):
        with torch.no_grad():
            shape = hiddenMaps.shape  #torch.Size([batch_size, feature_num, 26, 26])
            n_batches = shape[0]
            n_features = shape[1]
            k = 10
            
            top_k, _ = torch.topk(maxval, k, 0) 

            # Step 2: creating "drop" Array to be multiplied into featureMaps, dropping loser values
            drop = torch.where(maxval < top_k[k-1:k, :],  
                               torch.zeros((n_batches, n_features)).to(device), 
                               torch.ones((n_batches, n_features)).to(device))

        # To retain the graph, use .data to only modify the data of the tensor
        hiddenMaps.data = hiddenMaps.data * drop.reshape(n_batches, n_features, 1, 1).data