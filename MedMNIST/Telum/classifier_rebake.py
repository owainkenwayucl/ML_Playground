# This does not work yet - this is me trying to refactor all the code into a sensible design

import numpy
import sys
import argparse
from image_tools import process_images
from inference_tools import inference, compare_results
from multiprocess_tools import mp_inference
from multiprocessing import cpu_count
import json

classes_pathmnist = ('adipose','background','debris','lymphocytes','mucus','smooth muscle','normal colon mucosa','cancer-associated stroma','colorectal adenocarcinoma epithelium')

def main():
    parser = argparse.ArgumentParser(description="Image Classifier")
    parser.add_argument("--model",type=str, help="Model to use")
    parser.add_argument("images", nargs="+", type=str, help="A list of images to classify")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=512)

    args = parser.parse_args()

    classes = classes_pathmnist

    model =  "MedMNIST/medmnist_classifier_resnet18_pathmnist_55_20_32bit.so"
    if args.model != None:
        model = args.model

    filenames = args.images
    batch_size = args.batch_size

    nproc=cpu_count()

    stats = {}
    stats["nproc"] = nproc
    stats["model"] = model
    stats["batch size"] = batch_size

    #images, labels = process_images(filenames, classes)
    #matched = inference(images, model, classes)
    
    matched, labels, timing = mp_inference(filenames, classes, model, process_images, inference, nproc, batch_size)

    accuracy = compare_results(matched, labels)
    stats["accurace"] =  accuracy

    print(json.dumps(stats, indent=4))

if __name__ == "__main__":
    main()