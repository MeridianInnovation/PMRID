import torch

loss_fn_f1 = torch.nn.L1Loss()

import torch
import torch.nn.functional as F

# def loss_fn_f1(pred: torch.Tensor, label: torch.Tensor):
#     B = pred.shape[0]
#     ABS = torch.abs(pred - label)
#     Reshaped = torch.reshape(ABS, (B, -1))
#     L1 = Reshaped.mean(dim=1)
#     return L1.mean()

if __name__ == '__main__':
    # Set the seed 
    torch.manual_seed(0)
    # Define the input and target tensors
    input_tensor = torch.randn(3, 5, requires_grad=True)
    target_tensor = torch.randn(3, 5)
    print(input_tensor)
    print(target_tensor)

    # Compute the loss
    loss = loss_fn_f1(input_tensor, target_tensor)
    print('Loss:', loss.item())
