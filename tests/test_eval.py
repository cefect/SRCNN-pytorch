'''
Created on Dec. 4, 2023

@author: cef
'''


import pytest, os
import torch

from eval_calibrated import eval_model

from parameters import src_dir

#===============================================================================
# module variables
#===============================================================================
tdata = lambda x:os.path.join(src_dir, 'data', x)


#===============================================================================
# tests
#===============================================================================
@pytest.mark.parametrize('image_file',[tdata('butterfly_GT.bmp')])
@pytest.mark.parametrize('weights_file',[tdata('srcnn_x4.pth')])
def test_eval_calibrated(weights_file, image_file, tmp_path):
    eval_model(weights_file, image_file, out_dir=tmp_path)