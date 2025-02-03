# This does not work yet - this is me trying to refactor all the code into a sensible design

import numpy
import sys
import argparse
from image_tools import process_images

classes_pathmnist = ('adipose','background','debris','lymphocytes','mucus','smooth muscle','normal colon mucosa','cancer-associated stroma','colorectal adenocarcinoma epithelium')

def main():
    classes = classes_pathmnist

    filenames = ["MedMNIST/data/test/3153_0_adipose.png","MedMNIST/data/test/3154_0_adipose.png"] # test data

    images, labels = process_images(filenames, classes)

if __name__ == "__main__":
    main()