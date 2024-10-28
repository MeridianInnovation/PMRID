from model.model_torch import Network
# import pytorch now
import torch
from data.data_utils_torch import prepare_dataloaders
import os
import datetime
import pytz
from utils.utils import ssim_loss
from torch.utils.tensorboard import SummaryWriter
from utils.hyperparameters import Hyperparameters
import torch_optimizer as optim

# Define the training loop
def train_one_epoch(epoch_index, tb_writer, optimizer, model, 
                    training_loader, loss_fn, device):
    """
    Train the model for one epoch.

    Args:
        epoch_index: The index of the current epoch.
        tb_writer: The TensorBoard writer object.
        optimizer: The optimizer object.
        model: The model object.
        training_loader: The training data loader.
        loss_fn: The loss function.
        device: The device to use.

    Returns:
        The average loss per batch.
    """
    # Initialize the running loss. This will accumulate the loss per batch
    running_loss = 0
    # Initialize the last loss. This is used to report the loss per batch
    last_loss = 0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        # Move the data to the device
        inputs, labels = inputs.to(device), labels.to(device)

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
            running_loss = 0

    return last_loss

# Define the train function
def train(epochs, lr, checkpoints_folder, batch_size, optimizer_name, momentum=0.0):
    """
      Load the data, create the model, and train the model, saving the best model.

      Args:
        epochs: The number of epochs to train the model.
        lr: The learning rate.
        checkpoints_folder: The folder to save the model checkpoints.
        batch_size: The batch size.
        optimizer_name: The optimizer to use.
        momentum: The momentum for the optimizer.

      Returns:
        None
    """
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Create the model and set the device for the model
    model = Network().to(device)

    # Initialize the optimizer
    if optimizer_name == 'adam' or optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd' or optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optimizer_name == 'yogi' or optimizer_name == 'Yogi':
        # Yogi optimizer is not available in PyTorch, we use torch-optimizer
        # Yogi is a variant of Adam that uses a different update rule
        # It solves the problem of Adam's slow convergence or divergence when the learning rate is high
        optimizer = optim.Yogi(model.parameters(), lr=lr)
    else:
        raise ValueError('Invalid optimizer: {}'.format(optimizer))

    # Define the root path for the dataset
    root_dir = 'data'
    # Create the training and validation data loaders
    training_loader, validation_loader = prepare_dataloaders(root_dir, batch_size)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    hkt = pytz.timezone('Asia/Hong_Kong')
    timestamp = datetime.datetime.now(hkt)
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    
    epoch_number = 0
    best_vloss = float('inf') # Set the best validation loss to infinity

    # Define the loss function
    loss_fn = ssim_loss

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Trainning Phase
        # Make sure gradient tracking is on, and do a pass over the data
        # Set the model to training mode
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, optimizer, 
                                    loss_fn, model, training_loader, device)
        
        # Validation Phase
        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                # Move the data to the device
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
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
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            model_full_path = os.path.join(checkpoints_folder, model_path)
            torch.save(model.state_dict(), model_full_path)
            print(f'  Saved model at {model_path} with validation loss {best_vloss}')

        epoch_number += 1

if __name__ == "__main__":
    # Change the hyperpar ameters file name to the one you want to use
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