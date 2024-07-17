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

import time
import json
import sys
import os 
import math

timing = {}
timing["training"] = {}
timing["inference"] = {}

opts = poptorch.Options()

n_ipu = int(os.getenv("NUM_AVAILABLE_IPU", 4))
opts.replicationFactor(n_ipu)

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

dataset = "pathmnist"

mlbc = "multi-label, binary-class"

num_epochs = 10

# poptorch causes isssues with batch sizes which are not exact divisors
train_batch_size = 4
inference_batch_size = 4

if len(sys.argv) > 1:
    batch_size = int(sys.argv[1])
    train_batch_size = batch_size

remainder = 89996 % (train_batch_size * n_ipu)
if not remainder == 0:
    print(f">>> Warning: dropping {remainder} records from training set")

iremainder = 7180 % (inference_batch_size * n_ipu)
if not iremainder == 0:
    print(f">>> ERROR: {iremainder} records missing from test set due to running on {n_ipu} with an inference batch size of {inference_batch_size}.")
    sys.exit(1)

if len(sys.argv) > 2:
    inference_batch_size = int(sys.argv[2])
    # inference_batch_size = batch_size # do not allow batch setting on inference as it affects correctness.

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

train = data_class(split="train", transform=data_transform, download=True, size=224, mmap_mode='r')
test = data_class(split="test", transform=data_transform, download=True, size=224, mmap_mode='r')

train_dataloader = poptorch.DataLoader(opts, dataset=train, batch_size = train_batch_size, shuffle=True, drop_last=True, num_workers=20)
test_dataloader = poptorch.DataLoader(opts, dataset=test, batch_size = inference_batch_size, shuffle=False, drop_last=False)

print(train)
print(test)

class LossWrappedModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.resnet18(num_classes=num_classes)
        self.loss = criterion
    
    def forward(self, input, labels=None):
        out = self.model(input)

        if self.training:
            return out, self.loss(out, labels)
        return out


model = LossWrappedModel(num_classes=n_classes)

timing["training"]["start"] = time.time()
    
optimiser = poptorch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

poptorch_model = poptorch.trainingModel(model, options=opts, optimizer=optimiser)
model.train()

for epoch in range(num_epochs):
    epoch_start = time.time()
    for inputs, targets in tqdm.tqdm(train_dataloader):
        if task == mlbc:
            targets = targets.to(torch.float32)
        else:
            targets = targets.squeeze().long()
        _, loss = poptorch_model(inputs, targets)

    epoch_finish = time.time()
    timing["training"][f"epoch_{epoch}"] = epoch_finish - epoch_start

poptorch_model.detachFromDevice()

timing["training"]["finish"] = time.time()
timing["training"]["duration"] = timing["training"]["finish"] - timing["training"]["start"] 

timing["inference"]["start"] = time.time()

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

poptorch_model_inf.detachFromDevice()

timing["inference"]["finish"] = time.time()
timing["inference"]["duration"] = timing["inference"]["finish"] - timing["inference"]["start"] 

evaluator = medmnist.Evaluator(dataset, "test")
metrics = evaluator.evaluate(guess_score)

print(metrics)
print(json.dumps(timing, indent=4))
