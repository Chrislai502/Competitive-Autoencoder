# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:54:18 2021

@author: Chris-Lai
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
''' Parameters (Change Anything Here!) '''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Competitive_Segmentator(nn.Module):

    ''' Constructor __init__()
    Params:
        num_features - Previous output layer numfeatures
        prev_models - order as follows: [base_Model, stack_1, stack_2......., stack_n]
        prev_num_features - the number of previous layers if this is a Stacked Layer
    '''
    def __init__(self, prev_models = []):
        super().__init__()
        
        self.prev_models = prev_models
        self.num_prev_models = len(prev_models)
        self.prev_num_features = prev_models[-1].get_numfeatures()
        
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.prev_num_features, out_channels=self.prev_num_features/2, kernel_size=8, stride =2, padding = 9), # padding will decrease output size
            nn.ConvTranspose2d(in_channels=self.prev_num_features/2, out_channels = self.prev_num_features/4, kernel_size=8, stride =2, padding = 9),
            nn.ConvTranspose2d(in_channels=self.prev_num_features/4, out_channels =3, kernel_size=5, stride =1)
        )
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

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
        
        decoded = self.decoder(x)
        return decoded
    
    
    ''' bottleneck()
    - Forward x up till bottleneck layer of this model ONLY, and apply 2x2 max pooling
    - Returns the output x
    '''
    def bottleneck(self, x):
        return F.relu(self.pool(self.encoder(x)))
    
    ''' get_features()
    - Feeds Forward Input:x up until the bottleneck layer of this model ONLY
    - Applies Relu activation because spatial & lifetime is lifted (based on AliReza Paper)
    '''
    def get_features(self, x, prev_models = []):
        if len(prev_models) == 0:
            for model in prev_models:
                x = model.bottleneck(x)

        return F.relu(self.encoder(x))
    
    ''' get_numfeatures()
    - returns the number of features used in this layer/base_model
    '''
    def get_numfeatures(self):
        return self.num_features
