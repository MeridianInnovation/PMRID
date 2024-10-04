# src/training/train.py

import argparse
import os
import tensorflow as tf
from src.models.model import DenoiseNetwork
from src.data.data_utils import load_dataset
from src.utils.utils import loss_function

def parse_args():
    parser = argparse.ArgumentParser(description='Train the DenoiseNetwork.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id to use.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing training data.')
    parser.add_argument('--checkpoints_folder', type=str, default="saved_models/checkpoints", help='Directory to save checkpoints.')
    parser.add_argument('--pretrain_dir', type=str, help='Path to pre-trained model weights.')

    return parser.parse_args()

def train():
    args = parse_args()

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f'Using GPU: {physical_devices[int(args.gpu)]}')
        tf.config.experimental.set_memory_growth(physical_devices[int(args.gpu)], True)
    else:
        print('No GPU available, using CPU.')

    # Load Dataset
    dataset = load_dataset(args.data_dir, args.batch_size)

    # Initialize Model
    model = DenoiseNetwork()

    # Build Model (required for subclassed models)
    model.build(input_shape=(None, 160, 120, 1))
    model.summary()

    # Load Pre-trained Weights if Provided
    if args.pretrain_dir:
        model.load_weights(args.pretrain_dir)
        print(f'Loaded pre-trained weights from {args.pretrain_dir}')

    # Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss=loss_function, metrics=[loss_function])

    # Prepare Checkpoints Directory
    if not os.path.exists(args.checkpoints_folder):
        os.makedirs(args.checkpoints_folder)

    checkpoint_filepath = os.path.join(args.checkpoints_folder, 'best_model.h5')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # Start Training
    model.fit(
        dataset,
        epochs=args.epochs,
        callbacks=[checkpoint_callback]
    )

    # Save Final Model
    final_model_path = os.path.join(args.checkpoints_folder, 'final_model.h5')
    model.save_weights(final_model_path)
    print(f'Model weights saved to {final_model_path}')

if __name__ == '__main__':
    train()