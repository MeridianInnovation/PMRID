# data_loader_utils.py
from torch.utils.data import DataLoader
import os

def decode_image(file_path):
    """
      Function to decode and preprocess images.

      Args:
        file_path: The path to the image file.

      Returns:
        image: The preprocessed image in float32 format.
    """

def create_dataset(noisy_folder_dir, clean_folder_dir):
    """
      Function to create dataset from folder.

      Args:
        noisy_folder_dir: The directory containing the noisy images.
        clean_folder_dir: The directory containing the clean images.

      Returns:
        noisy_dataset: The dataset containing the noisy images.
        clean_dataset: The dataset containing the clean images.
    """


def prepare_dataloaders(root_dir, batch_size):
    """
      Prepare the training and validation data loaders.

      Args:
        root_dir: The root directory containing the dataset.
        batch_size: The batch size.
        create_dataset: The function to create the dataset.

      Returns:
        training_loader: The training data loader.
        validation_loader: The validation data loader
    """
    # Define the paths to the clean and noisy folders for training and validation
    train_noisy_folder_dir = os.path.join(root_dir, 'images_thermal_train_resized_noisy')
    train_clean_folder_dir = os.path.join(root_dir, 'images_thermal_train_resized_clean')
    val_noisy_folder_dir = os.path.join(root_dir, 'images_thermal_val_resized_noisy')
    val_clean_folder_dir = os.path.join(root_dir, 'images_thermal_val_resized_clean')

    # Create Dataset for training and validation
    train_noisy_dataset, train_clean_dataset = create_dataset(train_noisy_folder_dir, train_clean_folder_dir)
    val_noisy_dataset, val_clean_dataset = create_dataset(val_noisy_folder_dir, val_clean_folder_dir)

    # Create the training and validation data loaders
    # shuffle=True for training_loader, shuffle=False for validation_loader
    training_loader = DataLoader(list(zip(train_noisy_dataset, train_clean_dataset)), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(list(zip(val_noisy_dataset, val_clean_dataset)), batch_size=batch_size, shuffle=False)

    return training_loader, validation_loader