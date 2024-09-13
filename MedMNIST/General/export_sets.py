# This script exports the MednMNIST datasets into PNGs so they can be imported into non-PyTorch programs.

from PIL import Image

def ensure_directories(output_directories = ['train', 'test', 'val']):
    import os
    for a in output_directories:
        os.makedirs(a, exist_ok=True)

def export_sets():
    import torch, torchvision
    import medmnist
    
    dataset = "pathmnist"

    info = medmnist.INFO[dataset]
    task = info["task"]

    n_channels = info["n_channels"]
    n_classes = len(info["label"])
    classes = info["label"]

    data_class = getattr(medmnist, info["python_class"])

    dataset_names = ['test', 'val']
    #dataset_names = ['train', 'test', 'val']

    ensure_directories(dataset_names)

    datasets = {}

    #train = data_class(split="train", download=True, size=224, mmap_mode='r')
    test = data_class(split="test",  download=True, size=224, mmap_mode='r')
    val = data_class(split="val",  download=True, size=224, mmap_mode='r')

    #datasets["train"] = train
    datasets["test"] = test
    datasets["val"] = val

    for dataset in dataset_names:
        index = 0
        for element in datasets[dataset]:
            image = element[0]
            category_index = element[1][0]
            category = classes[int(element[1][0])]

            filename = f"{dataset}/{index}_{category_index}_{category}.png"
            print(f"Writing out {filename}")

            image.save(filename)
            index += 1


if __name__ == "__main__":
    export_sets()



