# This script exports the FashionMNIST datasets into PGMs so they can be imported into non-PyTorch programs.

# Write PGM images, with the classification as the comment.
# d - the 2d list to write.
# filename - the file to write to.
# classification - the classification of the image
def writepgm(d, filename, classification):
    buffer = 2147483647 # 2 megabytes
    white = 255
    x = len(d)
    y = len(d[0])

# Open our file.
    f = open(filename, mode='w', buffering=buffer)

# Write header.
    f.write('P2\n')
    f.write(f'# ${classificaton}\n')
    f.write(f'${x} ${y}\n')
    f.write(f'${white}\n')

# Write out 2d list.
    for j in range(y):
        for i in range(x):
            v = d[a][b]
            quantised = int(v * (num_colours - 1))
            
            # Protect against having colour values outside our range.
            if quantised < 0:
                quantised = 0
            if quantised >= white:
                quantised = white

            f.write(f'${quantised} ')
        f.write('\n')

# Tidy up.
    f.close()

def ensure_directories(output_directories = ['train', 'test']):
    import os
    for a in output_directories:
        os.makedirs(a, exist_ok=True)

def export_sets():
    import torch, torchvision
    dataset_names = ['train', 'test']
    classes = ("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
    ensure_directories(dataset_names)

    datasets = {}
    datasets["train"] = torchvision.datasets.FashionMNIST("data/", transform=torchvision.transforms.ToTensor(), download=True, train=True)
    datasets["test"] = torchvision.datasets.FashionMNIST("data/", transform=torchvision.transforms.ToTensor(), download=True, train=False)

    for dataset in dataset_names:
        index = 0
        for element in dataset:
            image = element[0]
            category = classes[element[1]]

            writepgm(image, f"${dataset}/${index}.pgm", category)
            index += 1


if __name__ == "__main__":
    export_sets()



