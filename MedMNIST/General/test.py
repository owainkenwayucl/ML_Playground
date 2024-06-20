# This is derived from this example notebook (https://github.com/MedMNIST/MedMNIST/blob/main/examples/getting_started.ipynb)
# Which is Apache Licensed and  Copyright 2020-2023 MedMNIST Team

# Converted to a script by Dr Owain Kenway, Summer 2024

import numpy
import torch
import torch.utils.data
import torchvision.transforms

import medmnist
import medmnist.INFO

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

dataset = "pathmnist"

num_epochs = 10

batch_size = 128

info = medmnist.INFO[dataset]
task = info["task"]

n_channels = info["n_channels"]
n_classes = len(info["label"])

data_class = getattr(medmnist, info["python_class"])

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std[0.5])
])

train = data_class(split="train", transform=data_transform, download=True)
test = data_class(split="test", transform=data_transform, download=True)

train_dataloader = torch.utils.data.DataLoader(dataset=train, batch_size = batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test, batch_size = batch_size, shuffle=False)

print(train)
print(test)