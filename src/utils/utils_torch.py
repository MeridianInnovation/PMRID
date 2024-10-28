import torch
import torch.nn.functional as F

# Loss functions

loss_fn_f1 = torch.nn.L1Loss()
# def loss_fn_f1(pred: torch.Tensor, label: torch.Tensor):
#     B = pred.shape[0]
#     ABS = torch.abs(pred - label)
#     Reshaped = torch.reshape(ABS, (B, -1))
#     L1 = Reshaped.mean(dim=1)
#     return L1.mean()

# Evaluation metrics

# Define the PSNR function
def calculate_psnr(output, target, max_pixel):
    """
      Calculate the Peak Signal-to-Noise Ratio (PSNR) for a batch of images.
    
      Args:
        output: The output image tensor.
        target: The target image tensor.
        max_pixel: The maximum pixel value.

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

# Define the SSIM function
def calculate_ssim(output, target):
    """
      Calculate the Structural Similarity Index (SSIM) for a batch of images.
    
      Args:
        output: The output image tensor.
        target: The target image tensor.

      Returns:
        The SSIM value.
    """ 
    # Calculate the SSIM
    ssim = pytorch_ssim.ssim(output, target)
    return ssim

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
