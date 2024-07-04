# This is derived from this example notebook (https://github.com/MedMNIST/MedMNIST/blob/main/examples/getting_started.ipynb)
# Which is Apache Licensed and  Copyright 2020-2023 MedMNIST Team

# Converted to a script by Dr Owain Kenway, Summer 2024

import numpy
import torch
import torch.nn
import torch.utils.data
import torch.optim
import torchvision.models
import torchvision.transforms
import tqdm

import habana_frameworks.torch.core as htcore
device = torch.device("hpu")

import medmnist

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

dataset = "pathmnist"

mlbc = "multi-label, binary-class"

num_epochs = 10

batch_size = 128

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

train = data_class(split="train", transform=data_transform, download=True, size=224, mmap_mode='r')
test = data_class(split="test", transform=data_transform, download=True, size=224, mmap_mode='r')

train_dataloader = torch.utils.data.DataLoader(dataset=train, batch_size = batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test, batch_size = batch_size, shuffle=False)

print(train)
print(test)



model = torchvision.models.resnet18(num_classes=n_classes)
model.to(device)

if task == mlbc:
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    criterion = torch.nn.CrossEntropyLoss()
    
optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

for epoch in range(num_epochs):
    model.train()

    for inputs, targets in tqdm.tqdm(train_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimiser.zero_grad()
        outputs = model(inputs)

        if task == mlbc:
            targets = targets.to(torch.float32)
        else:
            targets = targets.squeeze().long()

        loss = criterion(outputs, targets)
        loss.backward()
        htcore.mark_step()
        optimiser.step()
        htcore.mark_step()

model.eval()
guess_true = torch.tensor([])
guess_score = torch.tensor([])

with torch.no_grad():
    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs).to("cpu")
        htcore.mark_step()
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
