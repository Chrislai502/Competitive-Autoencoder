# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 10:32:54 2021
@source: https://gist.github.com/njanakiev/1932e0a450df6d121c05069d5f7d7d6f
@author: V510
"""

import os
from PIL import Image, ImageOps
import sys

working_path = "E:/Chris/Competitive-Autoencoder/"

sys.path.append(working_path)
os.chdir(working_path)
print("Working on path: ", os.getcwd())

brown = (165,42,42)
green = (42,130,42)

def concat_images(image_paths, size, shape=None, border_width = 1, color = 255, grey = False):
    
    assert border_width >= 0,"Border width HAVE to be a Natural Number"
    
    # Open images into array "images"
    size_1 = (size[0] + border_width*2, size[1] + border_width*2)
    width, height = size_1
    images = map(Image.open, image_paths)
    '''
    ## fitting the image to the input size given
    # images = [ImageOps.fit(image, size, Image.ANTIALIAS) 
    #           for image in images]
    ## Converting all images into greyscale
    # images = [ImageOps.grayscale(image)
    #                    for image in images]
    '''
    color = 255 if grey else (42,130,42)
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
            # break if reach the desired image value
            if idx >= len(image_paths):
                return ImageOps.expand(image,border=border_width,fill=color)
            image.paste(images[idx], offset)
    
    return image

# Get list of image paths
folder = 'Feature_plot_test'
num_features = 128
'''
## original source code
# count = 0
# for f in os.listdir(folder):
#     if f.endswith(str(count) + '.png'):
#         print(str(count) + '.png')
#         print(f)
#         print(count)
#     count +=1
#
# image_paths = [os.path.join(folder, f) 
#                for f in os.listdir(folder) if f.endswith('.png')]
'''
image_paths = []

for i in range(num_features):
    path = os.path.join(folder, "bottleneck_feature"+str(i)+".png") 
    image_paths.append(path)

# Create and save image grid (row, col)
image = concat_images(image_paths, (40, 40), (12, 11))
image.save('grid_features.jpg', 'JPEG')


#%%
import math
temp = math.sqrt(81)
temp1 = int(math.floor(temp))
shape = (temp1+1 if temp1 < temp else temp1, temp1)

print(shape)