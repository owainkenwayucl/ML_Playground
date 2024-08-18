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
import os

import pytorch_lightning

import medmnist

import onnx
import onnxruntime

mlbc = "multi-label, binary-class"

def detect_platform():
    num_acc = 1
    device = "cpu"
    # Set up devcies
    if torch.cuda.is_available():
        device = "cuda"
        num_acc = torch.cuda.device_count()
        for i in range(num_acc):
            device_name = torch.cuda.get_device_name(i)
            print(f"Detected Cuda Device: {device_name}")
        torch.set_float32_matmul_precision('high')
        print("Enabling TensorFloat32 cores.")
    else:
        try: 
            import poptorch
            num_acc = int(os.getenv("NUM_AVAILABLE_IPU", 1))
            print(f"Detected {n_ipu} Graphcore IPU(s)")
            device = "ipu"
        except:
            pass     

    return device, num_acc

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
        self.log("train_acc", accuracy, on_step=False, on_epoch=True, sync_dist=True)
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
        self.log("val_acc", accuracy, sync_dist=True)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        if self.task == mlbc:
            targets = targets.to(torch.float32)
        else:
            targets = targets.squeeze().long()

        loss = self.loss_module(outputs, targets)

        accuracy = (outputs.argmax(dim=-1) == targets).float().mean()
        self.log("test_acc", accuracy, sync_dist=True)

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
    print("Validation success")

def main():
    device, num_acc = detect_platform()
    print(f"Detected device config: {device}. {num_acc}")
    stats = {}
    # Define parameters
    dataset = "pathmnist"

    num_epochs = 10

    batch_size = 1024

    if len(sys.argv) > 1:        
        num_epochs = int(sys.argv[1])
        print(f" >>> Setting num_epochs to {num_epochs}")

    if len(sys.argv) > 2:
        batch_size = int(sys.argv[2])
        print(f" >>> Setting batch_size to {batch_size}")

    output_filename = f"medmnist_classifier_{dataset}_{num_epochs}"

    checkpoint_filename = f"{output_filename}.ckpt"
    onnx_filename = f"{output_filename}.onnx"
    weights_filename = f"{output_filename}.weights"
    json_filename = f"{output_filename}.json"

    stats["output_filename"] = output_filename
    stats["checkpoint_filename"] = checkpoint_filename
    stats["onnx_filename"] = onnx_filename
    stats["weights_filename"] = weights_filename
    stats["json_filename"] = json_filename

    trainer = pytorch_lightning.Trainer(max_epochs=num_epochs, accelerator=device, devices=num_acc)

    lr = 0.001

    train_dl, val_dl, test_dl, task, num_classes = generate_dataloaders(dataset, batch_size)

    model = Resnet_Classifier(device, task, lr, num_classes)

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    val_stats = trainer.validate(model=model, dataloaders=val_dl)
    test_stats = trainer.test(model=model, dataloaders=test_dl)

    stats["validation_stats"] = val_stats
    stats["test_stats"] = test_stats

    trainer.save_checkpoint(checkpoint_filename)
    torch.save(model.model.state_dict(), weights_filename)

    if trainer.global_rank == 0:
        write_onnx(model=model, filename=onnx_filename)

        log_data = json.dumps(stats, indent=4)
        print(log_data)

        with open(json_filename, "w") as lfh:
            lfh.write(log_data)


if __name__ == "__main__":
    main()
