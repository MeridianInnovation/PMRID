# data_loader_utils.py
from torch.utils.data import DataLoader
import os
from PIL import Image
import torchvision.transforms as transforms

def decode_image(file_path):
    """
      Function to decode and preprocess images.

      Args:
        file_path: The path to the image file.

      Returns:
        image: The preprocessed image.
    """
    image = Image.open(file_path)

    # Define the transformations to be applied to the image
    transform = transforms.Compose([
        transforms.Resize((120, 160)), # Resize the image to 120x160
        transforms.ToTensor(), # Convert the image to a PyTorch Tensor (C x H x W)
        # transforms.Normalize(mean=[0.5], std=[0.5]) # Normalize the image
    ])

    # Apply the transformations to the image
    image = transform(image)

    return image

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
    # Get the list of files in folders
    noisy_files = os.listdir(noisy_folder_dir)
    clean_files = os.listdir(clean_folder_dir)

    # Decode images
    noisy_dataset = [decode_image(os.path.join(noisy_folder_dir, file)) for file in noisy_files]
    clean_dataset = [decode_image(os.path.join(clean_folder_dir, file)) for file in clean_files]

    return noisy_dataset, clean_dataset


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
    train_noisy_folder_dir = os.path.join(root_dir, 'reduced_dataset_1_8', 'images_thermal_train_resized_noisy')
    train_clean_folder_dir = os.path.join(root_dir, 'reduced_dataset_1_8', 'images_thermal_train_resized_clean')
    val_noisy_folder_dir = os.path.join(root_dir, 'reduced_dataset_1_8', 'images_thermal_val_resized_noisy')
    val_clean_folder_dir = os.path.join(root_dir, 'reduced_dataset_1_8', 'images_thermal_val_resized_clean')

    # Create Dataset for training and validation
    train_noisy_dataset, train_clean_dataset = create_dataset(train_noisy_folder_dir, train_clean_folder_dir)
    val_noisy_dataset, val_clean_dataset = create_dataset(val_noisy_folder_dir, val_clean_folder_dir)

    # Create the training and validation data loaders
    # shuffle=True for training_loader, shuffle=False for validation_loader
    training_loader = DataLoader(list(zip(train_noisy_dataset, train_clean_dataset)), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(list(zip(val_noisy_dataset, val_clean_dataset)), batch_size=batch_size, shuffle=False)

    return training_loader, validation_loader

if __name__ == '__main__':
    # print current working directory
    print(os.getcwd())

    root_dir = 'data'
    batch_size = 128

    training_loader, validation_loader = prepare_dataloaders(root_dir, batch_size)

    print(f'Training data loader: {training_loader}')
    print(f'Validation data loader: {validation_loader}')

    # Print the first batch of data
    for noisy_images, clean_images in training_loader:
        print(f'Noisy images: {noisy_images.shape}')
        print(f'Clean images: {clean_images.shape}')
        break
    # Print the first batch of data
    for noisy_images, clean_images in validation_loader:
        print(f'Noisy images: {noisy_images.shape}')
        print(f'Clean images: {clean_images.shape}')
        break