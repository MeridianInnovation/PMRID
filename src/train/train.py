from model.model import DenoiseNetwork
import tensorflow as tf
from data.data_utils import prepare_dataloaders
import os
import datetime
import pytz
from utils.utils import ssim_loss

from src.utils.hyperparameters_torch import Hyperparameters

# Define the train function
def train(epochs, lr, gpu, checkpoints_folder, batch_size, optimizer_name, momentum=0.0):
    # Check GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    # print if we are using GPU
    print("Using GPU: ", gpu)

    # Define the root path for the dataset
    root_dir = 'data'
    # Create the training and validation data loaders
    train_dataset, val_dataset = prepare_dataloaders(root_dir, batch_size)

    # Initialize Optimizer
    if optimizer_name == 'adam' or optimizer_name == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_name == 'sgd' or optimizer_name == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum)
    else:
        raise ValueError('Invalid optimizer: {}'.format(optimizer_name))
    
    # Create Model
    model = DenoiseNetwork()
    # Compile Model
    model.compile(optimizer=opt, loss=ssim_loss, metrics=[ssim_loss])

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

    # define it for hk time
    hkt = pytz.timezone('Asia/Hong_Kong')
    # print when to starting time
    start_time = datetime.datetime.now(hkt)
    print(f'start training at {start_time} HKT...')

    # Train Model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        callbacks=[val_loss_checkpoint, train_loss_checkpoint],
        validation_data=val_dataset
    )

    # print when to finish
    finish_time = datetime.datetime.now(hkt)
    print(f'finish training at {finish_time} HKT.')


if __name__ == "__main__":
    # Parse the arguments
    hyperparams = Hyperparameters('hyperparameters_sample.yaml')

    # Call the train function with the parsed arguments
    train(
        epochs=hyperparams.epochs,
        lr=hyperparams.learning_rate,
        checkpoints_folder=hyperparams.checkpoints_folder,
        batch_size=hyperparams.batch_size,
        optimizer_name=hyperparams.optimizer,
        momentum=hyperparams.momentum,
    )