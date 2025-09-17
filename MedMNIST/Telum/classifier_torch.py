import numpy
import sys
import argparse
from image_tools import process_images
from inference_tools_torch import inference, compare_results
from multiprocess_tools import mp_inference, cpu_count
import json
import time

classes_pathmnist = ('adipose','background','debris','lymphocytes','mucus','smooth muscle','normal colon mucosa','cancer-associated stroma','colorectal adenocarcinoma epithelium')

def main():
    parser = argparse.ArgumentParser(description="Image Classifier")
    parser.add_argument("--model",type=str, help="Model to use")
    parser.add_argument("--weights",type=str, help="Weights to use")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=512)
    parser.add_argument("--mp", type=int, help="Number of worker processes", default=cpu_count())
    parser.add_argument("images", nargs="+", type=str, help="A list of images to classify")
    
    args = parser.parse_args()

    classes = classes_pathmnist

    weights =  "MedMNIST/medmnist_classifier_resnet152_pathmnist_25_20_32bit.weights"
    model = "resnet152"
    if args.model != None:
        model = args.model

    if args.weights != None:
        weights = args.weights

    filenames = args.images
    batch_size = args.batch_size

    nproc=args.mp

    stats = {}
    stats["file list"] = filenames
    stats["nproc"] = nproc
    stats["model"] = model
    stats["weights"] = weights
    stats["batch size"] = batch_size
   
    wall_start = time.time()
    matched, labels, timing = mp_inference(filenames, classes, (model, weights), process_images, inference, nproc, batch_size)
    wall_time = time.time() - wall_start

    accuracy = compare_results(matched, labels)
    stats["accuracy"] =  accuracy
    stats["timing"] = timing
    stats["timing"]["wall time"] = wall_time

    print(json.dumps(stats, indent=4))

if __name__ == "__main__":
    main()