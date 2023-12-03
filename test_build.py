'''
Created on Dec. 3, 2023

@author: cef
'''

import torch
import time
from tqdm import tqdm

print('start')
# Define a more complex computation: matrix multiplication
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

# Perform the computation on CPU
cpu_time = complex_computation('cpu')

# Check if CUDA is available
if torch.cuda.is_available():
    # Perform the computation on GPU
    gpu_time = complex_computation('cuda')

    # Print the difference in runtime
    print(f'Difference in runtime: {cpu_time - gpu_time} seconds')
else:
    print('CUDA is not available. The computation was only performed on CPU.')
