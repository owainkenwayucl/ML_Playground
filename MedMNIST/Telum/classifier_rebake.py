# This does not work yet - this is me trying to refactor all the code into a sensible design

import numpy
import sys
import argparse
from image_tools import process_images
from inference_tools import inference, compare_results
from multiprocess_tools import mp_inference

classes_pathmnist = ('adipose','background','debris','lymphocytes','mucus','smooth muscle','normal colon mucosa','cancer-associated stroma','colorectal adenocarcinoma epithelium')

def main():
    parser = argparse.ArgumentParser(description="Image Classifier")
    parser.add_argument("--model",type=str, help="Model to use")
    parser.add_argument("images", nargs="+", type=str, help="A list of images to classify")

    args = parser.parse_args()

    classes = classes_pathmnist

    model =  "MedMNIST/medmnist_classifier_resnet18_pathmnist_55_20_32bit.so"
    if args.model != None:
        model = args.model

    filenames = args.images

    #images, labels = process_images(filenames, classes)
    #matched = inference(images, model, classes)
    nproc=2
    matched, labels = mp_inference(filenames, classes, model, process_images, inference, nproc)

    print(matched)
    print(compare_results(matched, labels))

if __name__ == "__main__":
    main()