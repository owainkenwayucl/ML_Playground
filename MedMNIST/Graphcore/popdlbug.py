import torch
import poptorch
import torchvision

train_dataset = torchvision.datasets.FashionMNIST("data/", transform=torchvision.transforms.ToTensor(), download=True, train=True)

opts = poptorch.Options()

train_dataloader = poptorch.DataLoader(
    opts, train_dataset, batch_size=128, shuffle=True, num_workers=20, mode=poptorch.DataLoaderMode.AsyncRebatched
)

num = 0

for data, labels in train_dataloader:
    num += labels.shape[0]

print(f"Loaded {num} records from the FashionMNIST training set of 60,000 records.")