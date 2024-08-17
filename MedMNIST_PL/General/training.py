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

import onnx
import onnxruntime

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

def generate_dataloaders(dataset, batch_size):

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

    train_dataloader = torch.utils.data.DataLoader(dataset=train, batch_size = batch_size, shuffle=True, num_workers=71)
    val_dataloader = torch.utils.data.DataLoader(dataset=val, batch_size = batch_size, shuffle=False, num_workers=71)
    test_dataloader = torch.utils.data.DataLoader(dataset=test, batch_size = batch_size, shuffle=False, num_workers=71)

    #val_dataloader = None
    #test_dataloader = None

    print(train)

    return train_dataloader, val_dataloader, test_dataloader, task, n_classes

class Resnet_Classifier(pytorch_lightning.LightningModule):
    def __init__(self, device, task, lr, num_classes):
        super().__init__()
        self.model = torchvision.models.resnet18(num_classes=num_classes)
        self.task = task
        self.device_name = device
        self.lr = lr

        if task == mlbc:
            self.loss_module = torch.nn.BCEWithLogitsLoss()
        else:
            self.loss_module = torch.nn.CrossEntropyLoss()

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        if self.task == mlbc:
            targets = targets.to(torch.float32)
        else:
            targets = targets.squeeze().long()

        loss = self.loss_module(outputs, targets)

        accuracy = (outputs.argmax(dim=-1) == targets).float().mean()
        self.log("train_acc", accuracy, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        if self.task == mlbc:
            targets = targets.to(torch.float32)
        else:
            targets = targets.squeeze().long()

        loss = self.loss_module(outputs, targets)

        accuracy = (outputs.argmax(dim=-1) == targets).float().mean()
        self.log("val_acc", accuracy)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        if self.task == mlbc:
            targets = targets.to(torch.float32)
        else:
            targets = targets.squeeze().long()

        loss = self.loss_module(outputs, targets)

        accuracy = (outputs.argmax(dim=-1) == targets).float().mean()
        self.log("test_acc", accuracy)

    def configure_optimizers(self):
        optimiser = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        return optimiser

def write_onnx(model, filename):
    gibberish = torch.randn(1, 3, 224, 224, requires_grad=True)
    model_cpu = model.to("cpu")
    model_cpu.eval()
    torch_gibberish = model_cpu(gibberish)
    onnx_file = filename
    onnx_out_model = torch.onnx.export(model_cpu, 
                                gibberish,
                                onnx_file,
                                export_params = True,
                                input_names = ['input'],                       # the model's input names
                                output_names = ['output'],                     # the model's output names
                                dynamic_axes = {'input' : {0 : 'batch_size'},    # variable length axes
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


def main():
    device = detect_platform()

    # Define parameters
    dataset = "pathmnist"

    num_epochs = 10

    batch_size = 1024

    if len(sys.argv) > 1:
        num_epochs = int(sys.argv[1])

    if len(sys.argv) > 2:
        num_epochs = int(sys.argv[2])

    lr = 0.001

    train_dl, val_dl, test_dl, task, num_classes = generate_dataloaders(dataset, batch_size)

    model = Resnet_Classifier(device, task, lr, num_classes)

    trainer = pytorch_lightning.Trainer(max_epochs=num_epochs, strategy="ddp")
    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    trainer.validate(model=model, dataloaders=val_dl)
    trainer.test(model=model, dataloaders=test_dl)

    output_filename = f"medmnist_classifier_{dataset}_{num_epochs}"
    trainer.save_checkpoint(f"{output_filename}.ckpt")
    if trainer.global_rank == 0:
        temp_model = torch.load(f"{output_filename}.ckpt")
        write_onnx(model=temp_model, filename=f"{output_filename}.onnx")

if __name__ == "__main__":
    main()
