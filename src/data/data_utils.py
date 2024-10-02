import tensorflow as tf
import os

def load_dataset(data_dir, batch_size):
    # Assume data_dir contains 'clean' and 'noisy' subdirectories with images
    clean_images = tf.data.Dataset.list_files(os.path.join(data_dir, 'clean', '*.png'))
    noisy_images = tf.data.Dataset.list_files(os.path.join(data_dir, 'noisy', '*.png'))

    # Load and preprocess images
    def process_image(file_path):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_png(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image

    clean_dataset = clean_images.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
    noisy_dataset = noisy_images.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Zip datasets together and prepare batches
    dataset = tf.data.Dataset.zip((noisy_dataset, clean_dataset))
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset