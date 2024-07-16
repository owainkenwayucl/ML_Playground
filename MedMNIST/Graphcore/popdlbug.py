import torch
import poptorch
import torchvision
import torch.nn as nn
from tqdm import tqdm

import numpy

train_dataset = torchvision.datasets.FashionMNIST("data/", transform=torchvision.transforms.ToTensor(), download=True, train=True)

classes = ("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",)


train_dataloader = poptorch.DataLoader(
    opts, train_dataset, batch_size=128, shuffle=True, num_workers=20
)

opts = poptorch.Options()

num = 0

for data, labels in tqdm(train_dataloader, desc="batches", leave=False):
    print(lables.shape)