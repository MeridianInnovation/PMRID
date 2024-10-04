import tensorflow as tf
import numpy as np
import os
import sys

# Get the absolute path of the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the project root to the Python path
sys.path.insert(0, project_root)

# # Print the updated sys.path for debugging
# print(sys.path)

from src.models.model import DenoiseNetwork


# Instantiate the model
model = DenoiseNetwork()

# Create a dummy input with shape (batch_size, height, width, channels)
dummy_input = np.random.rand(1, 160, 120, 1).astype(np.float32)

# Run a forward pass
output = model(dummy_input)

# Check the output shape
print(f'Input shape: {dummy_input.shape}')
print(f'Output shape: {output.shape}')