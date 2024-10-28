import tensorflow as tf
import os

# Function to decode and preprocess images
# This function reads the image from the file path, decodes it, and preprocesses it
# The image is resized to 120x160 and converted to float32
# The image is assumed to be grayscale
# The function returns the preprocessed image in float32 format
def decode_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=1)  # Assuming grayscale images
    image = tf.cast(image, tf.float32)  # Convert to float32
    image.set_shape([120, 160, 1])
    return image

# Fuction to create dataset from folder
# This function takes the path to a folder containing images
# It creates a dataset from the images in the folder
# The function returns the dataset
def create_dataset(noisy_folder_dir, clean_folder_dir):
    # List files in the folder based on extension
    noisy_files = tf.data.Dataset.list_files(noisy_folder_dir + '/*.png')
    clean_files = tf.data.Dataset.list_files(clean_folder_dir + '/*.png')
    
    # Decode images
    noisy_dataset = noisy_files.map(lambda x: decode_image(x))
    clean_dataset = clean_files.map(lambda x: decode_image(x))

    return noisy_dataset, clean_dataset,

# Function to prepare data loaders
# This function prepares the training and validation data loaders
# It takes the root directory, batch size, as input
# It returns the training and validation data loaders
def prepare_dataloaders(root_dir, batch_size):
    # Define the paths to the clean and noisy folders for training and validation
    train_noisy_folder_dir = os.path.join(root_dir, 'reduced_dataset_1_8', 'images_thermal_train_resized_noisy')
    train_clean_folder_dir = os.path.join(root_dir, 'reduced_dataset_1_8', 'images_thermal_train_resized_clean')
    val_noisy_folder_dir = os.path.join(root_dir, 'reduced_dataset_1_8', 'images_thermal_val_resized_noisy')
    val_clean_folder_dir = os.path.join(root_dir, 'reduced_dataset_1_8', 'images_thermal_val_resized_clean')
    
    # Create Dataset for training and validation
    train_noisy_dataset, train_clean_dataset = create_dataset(train_noisy_folder_dir, train_clean_folder_dir)
    train_noisy_dataset = train_noisy_dataset.batch(batch_size)
    train_clean_dataset = train_clean_dataset.batch(batch_size)
    # validation dataset
    val_noisy_dataset, val_clean_dataset = create_dataset(val_noisy_folder_dir, val_clean_folder_dir)
    val_noisy_dataset = val_noisy_dataset.batch(batch_size)
    val_clean_dataset = val_clean_dataset.batch(batch_size)

    # zipping the datasets
    train_dataset = tf.data.Dataset.zip((train_noisy_dataset, train_clean_dataset))
    val_dataset = tf.data.Dataset.zip((val_noisy_dataset, val_clean_dataset))

    return train_dataset, val_dataset

if __name__ == '__main__':
    root_dir = 'data'
    batch_size = 128

    train_dataset, val_dataset = prepare_dataloaders(root_dir, batch_size)
    print(train_dataset)
    print(val_dataset)

    # Iterate over the first batch of the training dataset
    for noisy, clean in train_dataset.take(1):
        print(noisy.shape, clean.shape)
      
    # Iterate over the first batch of the validation dataset
    for noisy, clean in val_dataset.take(1):
        print(noisy.shape, clean.shape)
