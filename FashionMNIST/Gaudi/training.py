import torch
import torchvision
from tqdm import tqdm
import habana_frameworks.torch.core as htcore

train_dataset = torchvision.datasets.FashionMNIST("data/", transform=torchvision.transforms.ToTensor(), download=True, train=True)
test_dataset = torchvision.datasets.FashionMNIST("data/", transform=torchvision.transforms.ToTensor(), download=True, train=False)

classes = ("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",)

device = torch.device("hpu")

# Model structure from Graphcore example
class ClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 3)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(5, 12, 5)
        self.norm = torch.nn.GroupNorm(3, 12)
        self.fc1 = torch.nn.Linear(972, 100)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(100, 10)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.loss = torch.nn.NLLLoss()

    def forward(self, x, labels=None):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.norm(self.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.log_softmax(self.fc2(x))

        return x

model = ClassificationModel()

train_dataloader = torch.DataLoader(
    opts, train_dataset, batch_size=16, shuffle=True, num_workers=20
)

optimiser = torch.optim.Adam(model.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

model = model.to("hpu")
model = torch.compile(model,backend="hpu_backend")

epochs = 5
for epoch in tqdm(range(epochs), desc="epochs"):
    model.train()
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

torch.save(model.state_dict(), "fashion_classifier.pth")


print("\nDoing Inference... \n")

model = model.eval()

test_dataloader = torch.DataLoader(opts, test_dataset, batch_size=32, num_workers=10)

predictions, labels = [], []
for data, label in test_dataloader:
    predictions += model(data).data.max(dim=1).indices
    labels += label

for a in range(len(labels)):
    print(f"Label: {classes[labels[a]]} Prediction: {classes[predictions[a]]}")

print("\nThe Famous Ankle Boot Test!\n")

x, y = test_dataset[0][0], test_dataset[0][1]

from termshow import show, ANSI_COLOURS

print('Image:')
show(x[0], colours=ANSI_COLOURS)

print(f"Label: {classes[labels[0]]} Prediction: {classes[predictions[0]]}")


