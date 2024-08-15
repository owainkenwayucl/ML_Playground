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

import pytorch_lightning

import medmnist

mlbc = "multi-label, binary-class"

def detect_platform():
    # Set up devices
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        print(f"Detected Cuda Device: {device_name}")

        torch.set_float32_matmul_precision('high')
        print("Enabling TensorFloat32 cores.")
    else:
        device = torch.device("cpu")
    
    return device

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

def generate_dataloaders(dataset):

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
    val = test = data_class(split="val", transform=data_transform, download=True, size=224, mmap_mode='r')
    test = data_class(split="test", transform=data_transform, download=True, size=224, mmap_mode='r')

    train_dataloader = torch.utils.data.DataLoader(dataset=train, batch_size = train_batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val, batch_size = train_batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(dataset=test, batch_size = train_batch_size, shuffle=False)

    print(train)

    return train_dataloader, val_dataloader, test_dataloader, task

class Resnet_Classifier(pytorch_lightning.LightningModule):
    def __init__(self, device, task, lr):
        super().__init__
        self.model = torchvision.models.resnet18
        self.task = task
        self.device = device
        self.lr = lr

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = self.model(inputs)

        if self.task == mlbc:
            targets = targets.to(torch.float32)
        else:
            targets = targets.squeeze().long()

        loss = self.criterion(outputs, targets)
        return loss

    def configure_optimizers(self):
        optimiser = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        return optimiser



def main():
    device = detect_platform()

    # Define parameters
    dataset = "pathmnist"
    train_length = 89996
    inference_length = 7180

    num_epochs = 10

    train_batch_size = 32
    inference_batch_size = 32

    lr = 0.001

    train_dl, val_dl, test_dl, task = generate_dataloaders(dataset)

    res18 = Resnet_Classifier(device, task, lr)

    trainer = pytorch_lightning.Trainer()
    trainer.fit(model=res18, train_dataloaders=train_dl, max_epochs=num_epochs)

if __name__ == "__main__":
    main()