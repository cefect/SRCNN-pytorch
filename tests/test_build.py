'''
Created on Dec. 3, 2023

@author: cef
'''

import pytest
import torch
import time
from tqdm import tqdm


import torch
print(f'torch:{torch.__version__}')


import torchvision
print(f'torchvision:{torchvision.__version__}')

print(f"cuda available: {torch.cuda.is_available()}")

def complex_computation(device, dsize=int(1e4)):
    # Create random tensors
    x = torch.randn(dsize, dsize, device=device)
    y = torch.randn(dsize, dsize, device=device)

    # Start timer
    start_time = time.time()

    # Perform matrix multiplication
    for _ in tqdm(range(5), desc=f"Computing on {device}"):
        z = torch.matmul(x, y)

    # End timer
    end_time = time.time()

    # Return the elapsed time
    return end_time - start_time

# Define the devices to test
devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')

@pytest.mark.parametrize('device', devices)
def test_complex_computation(device):
    runtime = complex_computation(device)
    print(f'Runtime on {device}: {runtime} seconds')


 
