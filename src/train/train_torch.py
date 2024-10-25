from model.model_torch import Network
# import pytorch now
import torch
from torch.utils.data import DataLoader
from data.data_utils import create_dataset
import os
import datetime
import pytz
from utils.utils import ssim_loss
from torch.utils.tensorboard import SummaryWriter
from utils.hyperparameters import Hyperparameters

# Define the training loop
def train_one_epoch(epoch_index, tb_writer, optimizer, model, training_loader, loss_fn):
    """
    Train the model for one epoch.

    Args:
        epoch_index: The index of the current epoch.
        tb_writer: The TensorBoard writer object.
        optimizer: The optimizer object.
        model: The model object.
        training_loader: The training data loader.
        loss_fn: The loss function.

    Returns:
        The average loss per batch.
    """
    # Set the model to training mode
    running_loss = 0
    last_loss = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss

# Define the train function
def train(epochs, lr, gpu, checkpoints_folder, batch_size, optimizer_name, momentum=0.0):
    """
      Load the data, create the model, and train the model, saving the best model.

      Args:
        epochs: The number of epochs to train the model.
        lr: The learning rate.
        gpu: The GPU to use.
        checkpoints_folder: The folder to save the model checkpoints.
        batch_size: The batch size.
        optimizer_name: The optimizer to use.
        momentum: The momentum for the optimizer.

      Returns:
        None
    """
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
    
    # Create Dataset for training and validation
    train_clean_dataset, train_noisy_dataset = create_dataset(train_clean_folder_dir, train_noisy_folder_dir)
    val_clean_dataset, val_noisy_dataset = create_dataset(val_clean_folder_dir, val_noisy_folder_dir)
    
    # Create Dataloader for training and validation
    # Shuffle for training dataset, not shuffle for validation dataset
    training_loader = DataLoader(list(zip(train_noisy_dataset, train_clean_dataset)), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(list(zip(val_noisy_dataset, val_clean_dataset)), batch_size=batch_size, shuffle=False)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    hkt = pytz.timezone('Asia/Hong_Kong')
    timestamp = datetime.datetime.now(hkt)
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0

    best_vloss = 1_000_000

    # Create the model
    model = Network()

    # Initialize the optimizer
    if optimizer_name == 'adam' or optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd' or optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        raise ValueError('Invalid optimizer: {}'.format(optimizer))

    # Define the loss function
    loss_fn = ssim_loss

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)

        avg_loss = train_one_epoch(epoch_number, writer, optimizer, loss_fn, model, training_loader)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(model.state_dict(), checkpoints_folder)

        epoch_number += 1

if __name__ == "__main__":
    # Change the hyperpar ameters file name to the one you want to use
    hyperparams = Hyperparameters('hyperparameters_1016_0.yaml')

    # Call the train function with the parsed arguments
    train(
        epochs=hyperparams.epochs,
        lr=hyperparams.learning_rate,
        gpu=hyperparams.gpu,
        checkpoints_folder=hyperparams.checkpoints_folder,
        batch_size=hyperparams.batch_size,
        optimizer_name=hyperparams.optimizer,
        momentum=hyperparams.momentum,
    )