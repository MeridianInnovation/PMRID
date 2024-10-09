import tensorflow as tf

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
def create_dataset(clean_folder_dir, noisy_folder_dir):
    # List files in the folder based on extension
    clean_files = tf.data.Dataset.list_files(clean_folder_dir + '/*.png')
    noisy_files = tf.data.Dataset.list_files(noisy_folder_dir + '/*.png')
    
    # Decode images
    clean_dataset = clean_files.map(lambda x: decode_image(x))
    noisy_dataset = noisy_files.map(lambda x: decode_image(x))

    return clean_dataset, noisy_dataset
