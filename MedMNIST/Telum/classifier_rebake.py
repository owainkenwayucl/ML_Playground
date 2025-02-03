# This does not work yet - this is me trying to refactor all the code into a sensible design

from PIL import Image
import numpy
import sys
import argparse

classes_pathmnist = ('adipose','background','debris','lymphocytes','mucus','smooth muscle','normal colon mucosa','cancer-associated stroma','colorectal adenocarcinoma epithelium')

def extract_class_from_filename(filename, classes):
    c = "<not defined>"

    tokens = filename.split("_")
    if len(tokens) > 1:
        c = classes[int(tokens[1])]

    return c
    
def process_image(filename, classes):
    label = extract_class_from_filename(filename, classes)

    test_image_png = Image.open(filename)
    test_image_c = numpy.array(test_image_png, dtype=numpy.float32)
    test_image = numpy.moveaxis(test_image_c, 2, 0)

    ti_max = numpy.max(test_image)
    ti_min = numpy.min(test_image)

    # first put in range 0:1
    ti_range = 255.0

    test_image = numpy.divide(test_image, ti_range)

    # then approximate torchvision wtf transform
    test_image = numpy.subtract(test_image, 0.5)
    test_image = numpy.divide(test_image, 0.5)

    ti_max = numpy.max(test_image)
    ti_min = numpy.min(test_image)
    
    return numpy.copy(test_image, order='C'), label

def process_images(image_list, classes):
    labels = []
    images = []
    for a in image_list:
        i, l = process_image(a, classes)
        labels.append(l)
        images.append(i)

    return images, labels

def main():
    classes = classes_pathmnist

    filenames = ["MedMNIST/data/test/3153_0_adipose.png","MedMNIST/data/test/3154_0_adipose.png"] # test data

    images, labels = process_images(filenames, classes)



if __name__ == "__main__":
    main()