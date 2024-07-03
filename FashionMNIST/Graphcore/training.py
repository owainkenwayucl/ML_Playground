# This is based on https://docs.graphcore.ai/projects/tutorials/en/latest/pytorch/basics/README.htm

import torch
import poptorch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import onnx
import onnxruntime
import numpy

import time
import json

timing = {}
timing["training"] = {}
timing["inference"] = {}

train_dataset = torchvision.datasets.FashionMNIST("data/", transform=torchvision.transforms.ToTensor(), download=True, train=True)
test_dataset = torchvision.datasets.FashionMNIST("data/", transform=torchvision.transforms.ToTensor(), download=True, train=False)

classes = ("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",)

class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 12, 5)
        self.norm = nn.GroupNorm(3, 12)
        self.fc1 = nn.Linear(972, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def forward(self, x, labels=None):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.norm(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))
        # The model is responsible for the calculation
        # of the loss when using an IPU. We do it this way:
        if self.training:
            return x, self.loss(x, labels)
        return x


model = ClassificationModel()

timing["training"]["start"] = time.time()

model.train()

opts = poptorch.Options()

train_dataloader = poptorch.DataLoader(
    opts, train_dataset, batch_size=16, shuffle=True, num_workers=20
)

optimiser = poptorch.optim.Adam(model.parameters(), lr=0.001)

poptorch_model = poptorch.trainingModel(model, options=opts, optimizer=optimiser)

epochs = 5
for epoch in tqdm(range(epochs), desc="epochs"):
    epoch_start = time.time()
    total_loss = 0.0
    for data, labels in tqdm(train_dataloader, desc="batches", leave=False):
        output, loss = poptorch_model(data, labels)
        total_loss += loss
    epoch_finish = time.time()
    timing["training"][f"epoch_{epoch}"] = epoch_finish - epoch_start

poptorch_model.detachFromDevice()

timing["training"]["finish"] = time.time()
timing["training"]["duration"] = timing["training"]["finish"] - timing["training"]["start"] 

torch.save(model.state_dict(), "fashion_classifier.pth")

# Inference

print("\nDoing Inference... \n")
timing["inference"]["start"] = time.time()
model = model.eval()

poptorch_model_inf = poptorch.inferenceModel(model, options=opts)

test_dataloader = poptorch.DataLoader(opts, test_dataset, batch_size=32, num_workers=10)

predictions, labels = [], []
for data, label in test_dataloader:
    predictions += poptorch_model_inf(data).data.max(dim=1).indices
    labels += label

poptorch_model_inf.detachFromDevice()

timing["inference"]["finish"] = time.time()
timing["inference"]["duration"] = timing["inference"]["finish"] - timing["inference"]["start"] 

count_samples = len(labels)
correct = 0

for a in range(count_samples):
    if labels[a] == predictions[a]:
        correct += 1
    print(f"Label: {classes[labels[a]]} Prediction: {classes[predictions[a]]}")

percentage = 100 * (correct/count_samples)

print("\nThe Famous Ankle Boot Test!\n")

x, y = test_dataset[0][0], test_dataset[0][1]

from termshow import show, ANSI_COLOURS

print('Image:')
show(x[0], colours=ANSI_COLOURS)

print(f"Label: {classes[labels[0]]} Prediction: {classes[predictions[0]]}")
print(f"Prediction accuracy over training set: {percentage}% ")

# ONNX
gibberish = torch.randn(1, 1, 28, 28, requires_grad=True)
torch_gibberish = model(gibberish)
onnx_file = "fashion_classifier.onnx"
onnx_out_model = torch.onnx.export(model, 
                               gibberish,
                               onnx_file,
                               export_params=True,
                               input_names = ['input'],                       # the model's input names
                               output_names = ['output'],                     # the model's output names
                               dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                             'output' : {0 : 'batch_size'}})

print("Checking with ONNX")

onnx_model = onnx.load(onnx_file)

print("Checking with ONNX Runtime")


ort_session = onnxruntime.InferenceSession(onnx_file, providers=["CPUExecutionProvider"])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(gibberish)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
numpy.testing.assert_allclose(to_numpy(torch_gibberish), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")

print(json.dumps(timing, indent=4))