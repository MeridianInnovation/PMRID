import tensorflow as tf
import numpy as np
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