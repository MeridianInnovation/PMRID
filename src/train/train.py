from model.model import DenoiseNetwork
import tensorflow as tf
from data.data_utils import create_dataset
import argparse 
import os
from utils.utils import loss_function

# Define the train function
def train(epochs, lr, gpu, checkpoints_folder, batch_size):
    # Check GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    # print if we are using GPU
    print("Using GPU: ", gpu)

    # Define the root path for the dataset
    root_dir = 'data/'
    # Define the paths to the clean and noisy folders for training and validation
    train_clean_folder_dir = os.path.join(root_dir, 'images_thermal_train_resized_clean')
    train_noisy_folder_dir = os.path.join(root_dir, 'images_thermal_train_resized_noisy')
    val_clean_folder_dir = os.path.join(root_dir, 'images_thermal_val_resized_clean')
    val_noisy_folder_dir = os.path.join(root_dir, 'images_thermal_val_resized_noisy')
    
    # Get Data for training dataset
    train_clean_dataset, train_noisy_dataset = create_dataset(train_clean_folder_dir, train_noisy_folder_dir)
    train_clean_dataset = train_clean_dataset.batch(batch_size)
    train_noisy_dataset = train_noisy_dataset.batch(batch_size)

    # Get Data for validation dataset
    val_clean_dataset, val_noisy_dataset = create_dataset(val_clean_folder_dir, val_noisy_folder_dir)
    val_clean_dataset = val_clean_dataset.batch(batch_size)
    val_noisy_dataset = val_noisy_dataset.batch(batch_size)

    # Create Model
    model = DenoiseNetwork()

    # Compile Model
    opt = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=opt, loss=loss_function, metrics=[loss_function])

    # Define Checkpoint Callback
    val_loss_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoints_folder, 'best_model_val_loss_{epoch:02d}.weights.h5'),
        save_weights_only=True,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    train_loss_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoints_folder, 'best_model_train_loss_{epoch:02d}.weights.h5'),
        save_weights_only=True,
        monitor='loss',
        save_best_only=True,
        verbose=1
    )
    
    # zipping the datasets
    train_dataset = tf.data.Dataset.zip((train_clean_dataset, train_noisy_dataset))
    val_dataset = tf.data.Dataset.zip((val_clean_dataset, val_noisy_dataset))

    # Train Model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=[val_loss_checkpoint, train_loss_checkpoint],
        validation_data=val_dataset
    )

if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID to use')
    parser.add_argument('--checkpoints_folder', type=str, default="weights/ckpt", help='Folder to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')

    # Parse arguments
    config = parser.parse_args()

    # Call the train function with the parsed arguments
    train(
        epochs=config.epochs,
        lr=config.lr,
        gpu=config.gpu,
        checkpoints_folder=config.checkpoints_folder,
        batch_size=config.batch_size
    )