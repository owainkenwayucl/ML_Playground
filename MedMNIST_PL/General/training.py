import numpy
import torch
import torch.nn
import torch.utils.data
import torch.optim

import warnings

# Suppress Torchvision warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torchvision.models
import torchvision.transforms
warnings.filterwarnings("default", category=UserWarning)

import tqdm

import time
import json
import sys
import os
import argparse 

import pytorch_lightning

import medmnist

import onnx
import onnxruntime

mlbc = "multi-label, binary-class"

def detect_platform():
    num_acc = "auto"
    device = "auto"
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
            if num_acc <= 0:
                print(f"Detected Graphcore IPU(s), but NUM_AVAILABLE_IPU set to zero")
                device = "cpu"
            else: 
                print(f"Detected {num_acc} Graphcore IPU(s)")
                device = "ipu"
        except:
            pass     

    return device, num_acc


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
    def __init__(self, device, task, lr, base_model):
        super().__init__()
        self.model = base_model
        self.task = task
        self.device_name = device
        self.log_safe = True
        if (self.device_name == "ipu"):
            self.log_safe = False
            self.val_outputs = []
            self.test_outputs = []
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
        if self.log_safe:
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
        if self.log_safe:
            self.log("val_acc", accuracy, sync_dist=True)
        else:
            self.val_outputs.append(accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.model(inputs)

        if self.task == mlbc:
            targets = targets.to(torch.float32)
        else:
            targets = targets.squeeze().long()

        loss = self.loss_module(outputs, targets)

        accuracy = (outputs.argmax(dim=-1) == targets).float().mean()
        if self.log_safe:
            self.log("test_acc", accuracy, sync_dist=True)
        else:
            self.test_outputs.append(accuracy)
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self.log_safe:
            if str(self.device) == "cpu":
                self.log("val_acc", torch.stack(self.val_outputs).mean())
            self.val_outputs.clear()

    def on_test_epoch_end(self) -> None:
        if not self.log_safe:
            if str(self.device) == "cpu":
                self.log("test_acc", torch.stack(self.test_outputs).mean())
            self.test_outputs.clear()

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
    parser = argparse.ArgumentParser(description='Train model.')
    parser.add_argument('--epochs', metavar='epochs', type=int, help="Set the number of epochs.")
    parser.add_argument('--repeats', metavar='repeats', type=int, help="Set the number of repeats.")
    parser.add_argument('--batch-size', metavar='batchsize', type=int, help="Set the batch size.")
    parser.add_argument('--base-model', metavar='base_model', type=str, help="Model to use (default resnet18).")
    parser.add_argument('--half-precision', action='store_true', help="Train in 16 bit.")
    parser.add_argument('--lr', metavar='lr', type=float, help="Set learning rate.")
    args = parser.parse_args()

    print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

    device, num_acc = detect_platform()
    print(f"Detected device config: {device}:{num_acc}")
    stats = {}
    # Define parameters
    dataset = "pathmnist"

    num_epochs = 10

    batch_size = 1024
    if device == "ipu":
        batch_size = 2 # limited memory

    if args.epochs != None:
        num_epochs = args.epochs

    if args.batch_size != None:
        batch_size = args.batch_size

    if args.half_precision: 
        trainer = pytorch_lightning.Trainer(max_epochs=num_epochs, accelerator=device, devices=num_acc, precision=16)
    else: 
        trainer = pytorch_lightning.Trainer(max_epochs=num_epochs, accelerator=device, devices=num_acc)

    lr = 0.001
    if args.lr != None:
        lr = args.lr

    repeats = 1
    if args.repeats != None:
        repeats = args.repeats

    train_dl, val_dl, test_dl, task, num_classes = generate_dataloaders(dataset, batch_size)

    base_model = torchvision.models.resnet18(num_classes=num_classes)
    base_model_str = "resnet18"

    if args.base_model != None:
        if args.base_model == "resnet34":
            base_model_str = "resnet34"
            base_model = torchvision.models.resnet34(num_classes=num_classes)
        elif args.base_model == "resnet50":
            base_model_str = "resnet50"
            base_model = torchvision.models.resnet50(num_classes=num_classes)
        elif args.base_model == "resnet101":
            base_model_str = "resnet101"            
            base_model = torchvision.models.resne101(num_classes=num_classes)
        elif args.base_model == "resnet152":
            base_model_str = "resnet152"
            base_model = torchvision.models.resnet152(num_classes=num_classes)
      

    model = Resnet_Classifier(device, task, lr, base_model)

    prec_words = "32bit"

    for repeat in range(repeats):
        corrected_epochs = num_epochs * repeat
        print(f"Performing training iteration {repeat} of {repeats} for {corrected_epochs} epochs.")
        if args.half_precision: 
            print(f"Quantising 32 bit floats to 16 bit...")
            model = model.half()
            prec_words = "16bit"

        output_filename = f"medmnist_classifier_{base_model_str}_{dataset}_{corrected_epochs}_{repeats}_{prec_words}"

        checkpoint_filename = f"{output_filename}.ckpt"
        onnx_filename = f"{output_filename}.onnx"
        weights_filename = f"{output_filename}.weights"
        json_filename = f"{output_filename}.json"

        stats["output_filename"] = output_filename
        stats["checkpoint_filename"] = checkpoint_filename
        stats["onnx_filename"] = onnx_filename
        stats["weights_filename"] = weights_filename
        stats["json_filename"] = json_filename
        stats["device"] = device
        stats["num_accelerators"] = num_acc
        stats["num_epochs"] = corrected_epochs
        stats["repeats"] = repeats
        stats["batch_size"] = batch_size
        stats["lr"] = lr
        stats["precision"] = prec_words

        print(json.dumps(stats, indent=4))

        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        val_model = model
        val_trainer = trainer
        if device == "ipu":
            # Kludge for stat return bugs on Graphcore is to validate + test on CPU
            print(f"As we are running on Graphcore, validating on CPU...")
            if args.half_precision: 
                print(f"Up-casting 16 bit floats to 32 bit...")
                val_model = model.float()
            val_trainer = pytorch_lightning.Trainer(max_epochs=num_epochs, accelerator="cpu", devices=1)
        val_stats = val_trainer.validate(model=val_model, dataloaders=val_dl)
        test_stats = val_trainer.test(model=val_model, dataloaders=test_dl)

        stats["validation_stats"] = val_stats
        stats["test_stats"] = test_stats

        trainer.save_checkpoint(checkpoint_filename)
        torch.save(model.model.state_dict(), weights_filename)

        if trainer.global_rank == 0:
            write_onnx(model=val_model, filename=onnx_filename)

            log_data = json.dumps(stats, indent=4)
            print(log_data)

            with open(json_filename, "w") as lfh:
                lfh.write(log_data)


if __name__ == "__main__":
    main()
