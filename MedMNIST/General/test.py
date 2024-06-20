# This is derived from this example notebook (https://github.com/MedMNIST/MedMNIST/blob/main/examples/getting_started.ipynb)
# Which is Apache Licensed and  Copyright 2020-2023 MedMNIST Team

# Converted to a script by Dr Owain Kenway, Summer 2024

import numpy
import torch
import torch.nn
import torch.utils.data
import torchvision.transforms

import medmnist

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
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

train = data_class(split="train", transform=data_transform, download=True)
test = data_class(split="test", transform=data_transform, download=True)

train_dataloader = torch.utils.data.DataLoader(dataset=train, batch_size = batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test, batch_size = batch_size, shuffle=False)

print(train)
print(test)

class classification_model(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(classification_model, self).__init__()

        self.l1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )

        self.l2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, kernel_size=3),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.l3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )

        self.l4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )

        self.l5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.l6 = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, input_):
        input_ = self.l1(input_)
        input_ = self.l2(input_)
        input_ = self.l3(input_)
        input_ = self.l4(input_)
        input_ = self.l5(input_)
        input_ = input_.view(input_.size(0), -1)
        input_ = self.l6(input_)
        return input_

model = classification_model(in_channels = n_channels, num_classes=n_classes)