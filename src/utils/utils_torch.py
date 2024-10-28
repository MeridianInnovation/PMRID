import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Loss functions

# Define the L1 loss function
loss_fn_f1 = torch.nn.L1Loss()
# def loss_fn_f1(pred: torch.Tensor, label: torch.Tensor):
#     B = pred.shape[0]
#     ABS = torch.abs(pred - label)
#     Reshaped = torch.reshape(ABS, (B, -1))
#     L1 = Reshaped.mean(dim=1)
#     return L1.mean()

# Evaluation metrics

# Define the PSNR metric
def calculate_psnr_metric(output, target, max_pixel=255):
    """
      Calculate the Peak Signal-to-Noise Ratio (PSNR) for a batch of images.
    
      Args:
        output: The output image tensor.
        target: The target image tensor.
        max_pixel: The maximum pixel value. (Default: 255)

      Returns:
        The PSNR value.
    """
    # Calculate the MSE
    mse = F.mse_loss(output, target) 
    # Calculate the maximum pixel value
    # We assume the pixel values are in the range [0, 1]
    # but in the case of images, the pixel values are in the range [0, 255]
    # Calculate the PSNR
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

# Define the SSIM metric
def calculate_ssim_metric(output, target, data_range=255):
    """
      Calculate the Structural Similarity Index (SSIM) for a batch of images.
    
      Args:
        output: The output image tensor.
        target: The target image tensor.
        data_range: The range of the pixel values (Default: 255)

      Returns:
        The SSIM value.
    """ 
    # Calculate the SSIM
    # Change from tensor to numpy
    output = output.cpu().detach().numpy()
    target = target.cpu().detach().numpy()
    # create a list to store the ssim values
    ssim_values = []
    # iterate over the batch of images (output and target)
    for o, t in zip(output, target):
        # calculate the ssim value for each pair of images
        # excluding the channel dimension
        o = np.squeeze(o, axis=0)
        t = np.squeeze(t, axis=0)
        ssim_value = ssim(o, t, data_range=data_range, win_size=11)
        # append the ssim value to the list
        ssim_values.append(ssim_value)
    
    return np.mean(ssim_values)

# Learning rate scheduler


if __name__ == '__main__':
    # Set the seed 
    torch.manual_seed(0)
    # Define the input and target tensors
    input_tensor = torch.randn(64, 1, 120, 160, requires_grad=True)
    target_tensor = input_tensor.clone()
    # print(input_tensor)
    print('size:', input_tensor.size())
    # print(target_tensor)
    print('size:', target_tensor.size())

    # Compute the loss
    loss = loss_fn_f1(input_tensor, target_tensor)
    print('Loss:', loss.item())

    # Compute the PSNR
    psnr = calculate_psnr_metric(input_tensor, target_tensor)
    print('PSNR:', psnr.item())
    ssim = calculate_ssim_metric(input_tensor, target_tensor)
    print('SSIM:', ssim)
