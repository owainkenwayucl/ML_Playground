from PyRuntime import OMExecutionSession
import json
import numpy as np
import time

model = "MedMNIST/medmnist_classifier_pathmnist_30_PL.so"

convert_types = {"f32":"float32",
                 "f16":"float16"}

def inference(image_data):
    setup_start = time.time()
    session = OMExecutionSession(model)
    input_signature_json = json.loads(session.input_signature())
    signature = input_signature_json[0]
    input_type = signature["type"]

    images = image_data[0][np.newaxis,np.newaxis,...].astype(convert_types[input_type])
    
    if (len(image_data) > 1):
        for a in range(1, len(image_data)):
            image = image_data[a][np.newaxis,np.newaxis,...].astype(convert_types[input_type])
            images = np.concatenate((images, image), axis=0)
    imageset = []
    imageset.append(images)
    setup_stop = time.time()
    print(f"Time in setup: {setup_stop - setup_start}")
    output = {}
    inf_start = time.time()
    output["output"] = session.run(imageset)
    inf_stop = time.time()
    print(f"Time in inference: {inf_stop - inf_start}")

    return output


def process_image(filename):
    from PIL import Image

    test_image_png = Image.open(filename)
    test_image = numpy.array(test_image_png, dtype=numpy.float32)

    ti_max = numpy.max(test_image)
    ti_min = numpy.min(test_image)

    print(f"Image max {ti_max}, image min {ti_min}")

    ti_range = ti_max - ti_min

    test_image = numpy.subtract(test_image, ti_min)
    test_image = numpy.divide(test_image, ti_range)

    ti_max = numpy.max(test_image)
    ti_min = numpy.min(test_image)

    print(f"Normalised image max {ti_max}, image min {ti_min}")

    return test_image

def main():
    import sys

    iname = ["test.png"]

    classes = ('adipose','background','debris','lymphocytes','mucus','smooth muscle','normal colon mucosa','cancer-associated stroma','colorectal adenocarcinoma epithelium')

    if len(sys.argv) > 1:
        iname = []
        for a in range(1,len(sys.argv)):
            iname.append(sys.argv[a])

    labels = []
    images = []
    for a in iname:
        image = process_image(a)
        images.append(image)
        c = "<not defined>"

        tokens = iname.split("_")
        if len(tokens) > 1:
            c = classes[int(tokens[1])]
        labels.append(c)

    
    n = len(labels)
    
    start = time.time()
    results = inference(images)
    stop = time.time()

    #print(results)
    correct = 0

    for a in range(n):
        c = labels[a]
        r = classes[np.argmax(results['output'][0][a])]
        if (c == r):
            correct = correct + 1
        #print(f"Expected: {c}\nPredicted: {r}")

    
    print(f"Percentage correct: {100*(correct/n)}%")
    print(f"Time taken for {n} inferences: {stop - start} seconds")

if __name__ == "__main__":
    main()
