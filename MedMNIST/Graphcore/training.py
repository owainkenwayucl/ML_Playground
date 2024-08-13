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

import onnx
import onnxruntime

import time
import json
import sys
import os 
import math

timing = {}
timing["training"] = {}
timing["inference"] = {}

opts = poptorch.Options()
opts_inf = poptorch.Options()
opts_inf.replicationFactor(1) # use one IPU for inference

n_ipu = int(os.getenv("NUM_AVAILABLE_IPU", 16))
opts.replicationFactor(n_ipu)

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

dataset = "pathmnist"
train_length = 89996
inference_length = 7180

mlbc = "multi-label, binary-class"

num_epochs = 20

# poptorch causes isssues with batch sizes which are not exact divisors
train_batch_size = 1
inference_batch_size = 4

if len(sys.argv) > 1:
    batch_size = int(sys.argv[1])
    train_batch_size = batch_size

remainder = train_length % (train_batch_size * n_ipu)
actual_train_length = train_length - remainder
if not remainder == 0:
    print(f">>> Warning: dropping {remainder} records from training set")

    

if len(sys.argv) > 2:
    inference_batch_size = int(sys.argv[2])

iremainder = inference_length % inference_batch_size
if not iremainder == 0:
    print(f">>> ERROR: {iremainder} records missing from test set due to running on {n_ipu} IPUs with an inference batch size of {inference_batch_size}.")
    sys.exit(1)
    # inference_batch_size = batch_size # do not allow batch setting on inference as it affects correctness.

if len(sys.argv) > 3:
    num_epochs = int(sys.argv[3])

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

train_temp = data_class(split="train", transform=data_transform, download=True, size=224, mmap_mode='r')
indices = torch.arange(actual_train_length)
train = torch.utils.data.Subset(train_temp, indices)

test = data_class(split="test", transform=data_transform, download=True, size=224, mmap_mode='r')

train_dataloader = poptorch.DataLoader(opts, dataset=train, batch_size = train_batch_size, shuffle=True, drop_last=False, num_workers=20)
test_dataloader = poptorch.DataLoader(opts_inf, dataset=test, batch_size = inference_batch_size, shuffle=False, drop_last=False)

print(train_temp)
print(test)

class LossWrappedModel(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.resnet50(num_classes=num_classes)
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

poptorch_model_inf = poptorch.inferenceModel(model, options=opts_inf)

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

# ONNX
gibberish = torch.randn(1, 3, 224, 224, requires_grad=True)
torch_gibberish = model(gibberish)
onnx_file = f"medical_classifier_{num_epochs}.onnx"
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

print(" --- CUT HERE --- ")

metrics["onnx filename"] = onnx_file
log_filename = f"{onnx_file}.log"
timing["metrics"] = metrics

log_data = json.dumps(timing, indent=4)
print(log_data)

with open(log_filename, "w") as lfh:
    lfh.write(log_data)

