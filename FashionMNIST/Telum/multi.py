from PyRuntime import OMExecutionSession
import json
import numpy as np
model = "fashion_MNIST/fashion_classifier.so"

convert_types = {"f32":"float32",
                 "f16":"float16"}

def inference(image_data):
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
    output = {}
    output["output"] = session.run(imageset)

    return output

def main():
    import sys
    import time
    from imageio import readpgm
    from termshow import show, ANSI_COLOURS

    iname = ["test.pgm"]

    classes = ("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",)

    if len(sys.argv) > 1:
        iname = []
        for a in range(1,len(sys.argv)):
            iname.append(sys.argv[a])

    labels = []
    images = []
    for a in iname:
        c, image = readpgm(a)
        images.append(image)
        labels.append(c)
        #print(f"Loaded image: {a} Label: {c}")
        #show(image, ANSI_COLOURS)
    
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
