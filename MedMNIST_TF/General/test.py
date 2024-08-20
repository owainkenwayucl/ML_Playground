# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import termshow
import sys
import time 
import argparse

# MedMNIST
import medmnist

def load_data(dataset):
    info = medmnist.INFO[dataset]
    task = info["task"]

    n_channels = info["n_channels"]
    n_classes = len(info["label"])

    data_class = getattr(medmnist, info["python_class"])

    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train = data_class(split="train", transform=data_transform, download=True)
    test = data_class(split="test", transform=data_transform, download=True)

    return train, test


def main():
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--epochs', metavar='epochs', type=int, help="Set the number of epochs.")

    args = parser.parse_args()

    print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")
    stats = {}

    # Define parameters
    dataset = "pathmnist"

    num_epochs = 10

    if args.epochs != None:
        num_epochs = args.epochs



if __name__ == "__main__":
    main()