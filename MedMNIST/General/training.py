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

import time
import json
import sys

import onnx
import onnxruntime

timing = {}
timing["training"] = {}
timing["inference"] = {}


if torch.cuda.is_available():
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    print(f"Detected Cuda Device: {device_name}")

    torch.set_float32_matmul_precision('high')
    print("Enabling TensorFloat32 cores.")
else:
    device = torch.device("cpu")

import medmnist

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

dataset = "pathmnist"
train_length = 89996
inference_length = 7180

mlbc = "multi-label, binary-class"

num_epochs = 10

# poptorch causes isssues with batch sizes which are not exact divisors - need to be fair in other platforms
train_batch_size = 149
inference_batch_size = 359

if len(sys.argv) > 1:
    batch_size = int(sys.argv[1])
    train_batch_size = batch_size
    inference_batch_size = batch_size

remainder = train_length % train_batch_size
actual_train_length = train_length - remainder
if not remainder == 0:
    print(f">>> Warning: dropping {remainder} records from training set")

if len(sys.argv) > 2:
    inference_batch_size = int(sys.argv[2])

if len(sys.argv) > 3:
    actual_train_length = int(sys.argv[3])
    remainder = train_length - actual_train_length
    print(f">>> Warning: dropping {remainder} records from training set as requested")
    
if len(sys.argv) > 4:
    num_epochs = int(sys.argv[4])


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

train_temp = data_class(split="train", transform=data_transform, download=True, size=224, mmap_mode='r')
indices = torch.arange(actual_train_length)
train = torch.utils.data.Subset(train_temp, indices)

test = data_class(split="test", transform=data_transform, download=True, size=224, mmap_mode='r')

train_dataloader = torch.utils.data.DataLoader(dataset=train, batch_size = train_batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset=test, batch_size = train_batch_size, shuffle=False)

print(train_temp)
print(test)



model = torchvision.models.resnet18(num_classes=n_classes)

timing["training"]["start"] = time.time()

model.to(device)

if task == mlbc:
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    criterion = torch.nn.CrossEntropyLoss()
    
optimiser = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)


for epoch in range(num_epochs):
    epoch_start = time.time()
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
        optimiser.step()
    epoch_finish = time.time()
    timing["training"][f"epoch_{epoch}"] = epoch_finish - epoch_start

timing["training"]["finish"] = time.time()
timing["training"]["duration"] = timing["training"]["finish"] - timing["training"]["start"] 

timing["inference"]["start"] = time.time()

model.eval()
guess_true = torch.tensor([])
guess_score = torch.tensor([])

with torch.no_grad():
    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs).to("cpu")
        if task == mlbc:
            targets = targets.to(torch.float32)
        else:
            targets = targets.squeeze().long()
            targets = targets.float().resize_(len(targets), 1)

        outputs = outputs.softmax(dim=-2)

        guess_true = torch.cat((guess_true, targets),0)
        guess_score = torch.cat((guess_score, outputs),0)

timing["inference"]["finish"] = time.time()
timing["inference"]["duration"] = timing["inference"]["finish"] - timing["inference"]["start"] 

evaluator = medmnist.Evaluator(dataset, "test", size=224)
metrics = evaluator.evaluate(guess_score)

# ONNX
gibberish = torch.randn(1, 3, 224, 224, requires_grad=True)
model_cpu = model.to("cpu")
torch_gibberish = model_cpu(gibberish)
onnx_file = f"medical_classifier_{num_epochs}.onnx"
onnx_out_model = torch.onnx.export(model_cpu, 
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

print(metrics)

print(" --- CUT HERE --- ")

metrics_ = {}
info = {}
info["onnx filename"] = onnx_file
metrics_["AUC"] = metrics[0]
metrics_["Accuracy"] = metrics[1]
log_filename = f"{onnx_file}.log"
info["metrics"] = metrics_
info["timing"] = timing

log_data = json.dumps(info, indent=4)
print(log_data)

with open(log_filename, "w") as lfh:
    lfh.write(log_data)

