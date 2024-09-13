# This script exports the MednMNIST datasets into PNGs so they can be imported into non-PyTorch programs.

from PIL import Image

def ensure_directories(output_directories = ['train', 'test', 'val']):
    import os
    for a in output_directories:
        os.makedirs(a, exist_ok=True)

def str_to_filename(str):
    return str.replace(" ", "_").replace("/", "_")

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

    dataset_names = ['train', 'test', 'val']

    ensure_directories(dataset_names)

    datasets = {}

    datasets["train"] = data_class(split="train", download=True, size=224, mmap_mode='r')
    datasets["test"] = data_class(split="test",  download=True, size=224, mmap_mode='r')
    datasets["val"] = data_class(split="val",  download=True, size=224, mmap_mode='r')

    for dataset in dataset_names:
        index = 0
        for element in datasets[dataset]:
            image = element[0]
            category_index = element[1][0]
            category = classes[str(element[1][0])]

            filename = f"{dataset}/{index}_{category_index}_{str_to_filename(category)}.png"
            print(f"Writing out {filename}")

            image.save(filename)
            index += 1


if __name__ == "__main__":
    export_sets()



