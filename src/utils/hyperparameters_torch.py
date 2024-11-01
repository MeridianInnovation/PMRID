
import yaml
import os

# This can be used by both pytorch and tensorflow

# Define a class to store hyperparameters
# This class will be used to store hyperparameters for the training process
class Hyperparameters:
    # def __init__(self, learning_rate=1e-5, batch_size=64, epochs=20, optimizer='SGD', momentum=0.0):
    #     self.learning_rate = learning_rate
    #     self.batch_size = batch_size
    #     self.epochs = epochs
    #     self.optimizer = optimizer
    #     self.momentum = momentum

    def __init__(self, config_file):
        self.load_hyperparameters(config_file)

    def load_hyperparameters(self, config_file):
        """ Load hyperparameters from a config file """
        path_to_config = os.path.join(os.getcwd(), 'configs', config_file)
        with open(path_to_config, 'r') as file:
            config = yaml.safe_load(file)
        
        # Set the hyperparameters
        self.learning_rate = config['learning_rate']
        # convert the learning rate to float
        self.learning_rate = float(self.learning_rate)
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.optimizer = config['optimizer']
        self.momentum = config['momentum']
        self.checkpoints_folder = config['checkpoints_folder']

    def print_params(self):
        print(f"Learning Rate: {self.learning_rate}")
        # log the type of learning rate
        print(type(self.learning_rate))
        print(f"Batch Size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Optimizer: {self.optimizer}")
        print(f"Momentum: {self.momentum}")
        print(f"Checkpoints Folder: {self.checkpoints_folder}")

if __name__ == "__main__":
    # print(os.getcwd()
    hyperparams = Hyperparameters('hyperparameters_sample.yaml')
    hyperparams.print_params()
