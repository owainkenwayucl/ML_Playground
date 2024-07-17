# This is derived from this example notebook (https://github.com/MedMNIST/MedMNIST/blob/main/examples/getting_started.ipynb)
# Which is Apache Licensed and  Copyright 2020-2023 MedMNIST Team

# Converted to a script by Dr Owain Kenway, Summer 2024

import numpy
import torch
import torch.nn
import torch.utils.data
import torch.optim
import torchvision.transforms
import poptorch
import tqdm

import medmnist

opts = poptorch.Options()

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

dataset = "pathmnist"

mlbc = "multi-label, binary-class"

num_epochs = 10

batch_size = 256 
lr = 0.001

info = medmnist.INFO[dataset]
task = info["task"]

n_channels = info["n_channels"]
n_classes = len(info["label"])

data_class = getattr(medmnist, info["python_class"])

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

if task == mlbc:
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    criterion = torch.nn.CrossEntropyLoss()

train = data_class(split="train", transform=data_transform, download=True)
test = data_class(split="test", transform=data_transform, download=True)

train_dataloader = poptorch.DataLoader(opts, dataset=train, batch_size = batch_size, shuffle=True, num_workers=20, mode=poptorch.DataLoaderMode.AsyncRebatched)
test_dataloader = poptorch.DataLoader(opts, dataset=test, batch_size = batch_size, shuffle=False, mode=poptorch.DataLoaderMode.AsyncRebatched)

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
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
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
        self.loss = criterion

    def forward(self, input_, labels=None):
        input_ = self.l1(input_)
        input_ = self.l2(input_)
        input_ = self.l3(input_)
        input_ = self.l4(input_)
        input_ = self.l5(input_)
        input_ = input_.view(input_.size(0), -1)
        input_ = self.l6(input_)
        if self.training:
            return input_, self.loss(input_, labels)
        return input_

model = classification_model(in_channels = n_channels, num_classes=n_classes)

    
optimiser = poptorch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

poptorch_model = poptorch.trainingModel(model, options=opts, optimizer=optimiser)
model.train()

for epoch in range(num_epochs):
    for inputs, targets in tqdm.tqdm(train_dataloader):
        if task == mlbc:
            targets = targets.to(torch.float32)
        else:
            targets = targets.squeeze().long()
        _, loss = poptorch_model(inputs, targets)

model.eval()
guess_true = torch.tensor([])
guess_score = torch.tensor([])

poptorch_model_inf = poptorch.inferenceModel(model, options=opts)

with torch.no_grad():
    for inputs, targets in test_dataloader:
        outputs = poptorch_model_inf(inputs)

        if task == mlbc:
            targets = targets.to(torch.float32)
        else:
            targets = targets.squeeze().long()
            targets = targets.float().resize_(len(targets), 1)

        outputs = outputs.softmax(dim=-2)

        guess_true = torch.cat((guess_true, targets),0)
        guess_score = torch.cat((guess_score, outputs),0)

evaluator = medmnist.Evaluator(dataset, "test")
metrics = evaluator.evaluate(guess_score)

print(metrics)

