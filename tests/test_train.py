'''
Created on Dec. 4, 2023

@author: cef
'''



import pytest, os
import torch

from train import train_model

#===============================================================================
# module variables
#===============================================================================
tdata = lambda x:os.path.join(r'l:\10_IO\2307_super\ins\SRCNN\set5', x)


#===============================================================================
# tests
#===============================================================================
@pytest.mark.parametrize('train_file',[tdata('91-image_x3.h5')])
@pytest.mark.parametrize('eval_file',[tdata('Set5_x3.h5')])
def test_train_model(train_file, eval_file, tmp_path):
    train_model(train_file, eval_file, out_dir=tmp_path, num_epochs=2 )