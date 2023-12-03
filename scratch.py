'''
Created on Dec. 3, 2023

@author: cef
'''

import h5py, os
import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from datasets import EvalDataset

from utils import convert_ycbcr_to_rgb

def extract_images(
        data_file,
        out_dir = os.getcwd(),
        group_key='lr'
        ):
    """load the h5py datafile and extract each image from the group
    
    
    NOTE: the data format is challenging to work with...
    the images may have originally been RGB,
    but the provided .h5 files only contain the y-channel of YCbCr
    """
    assert os.path.exists(data_file)
    
    with h5py.File(data_file, 'r') as h5_file:
    
            # Access the 'lr' group
        lr_group = h5_file[group_key]
        
        # Choose the first dataset in the 'lr' group
        for i in range(len(lr_group)):
            lr_image = lr_group[str(i)]
            
            # Convert the dataset to a numpy array
            lr_array = np.array(lr_image)
            
            convert_ycbcr_to_rgb(lr_array)
            
            # Normalize the array to 0-255
            lr_array = ((lr_array - lr_array.min()) * (255 / (lr_array.max() - lr_array.min()))).astype(np.uint8)
            
            # Convert the array to an image
            img = Image.fromarray(lr_array)
            img.save(os.path.join(out_dir, f'{group_key}_{i}.png'))
 
 
        
    print(f'saved images to \n    {out_dir}')
        

if __name__ == '__main__':
    
    data_file = r'l:\10_IO\2307_super\ins\SRCNN\set5\Set5_x3.h5'
    
    extract_images(data_file, out_dir=r'l:\10_IO\2307_super\outs\SRCNN\set5_x3')