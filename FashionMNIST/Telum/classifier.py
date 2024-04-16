from PyRuntime import OMExecutionSession
import json
import numpy as np
model = "fashion_MNIST/fashion_classifier.so"

convert_types = {"f32":"float32",
                 "f16":"float16"}

def inference(image):
    session = OMExecutionSession(model)
    input_signature_json = json.loads(session.input_signature())
    signature = input_signature_json[0]
    input_type = signature["type"]

    image = image[np.newaxis,np.newaxis,...].astype(convert_types[input_type])
    imageset = []
    imageset.append(image)
    output = {}
    output["output"] = session.run(imageset)

    return output

def main():
    import sys
    from imageio import readpgm
    from termshow import show, ANSI_COLOURS

    iname = "test.pgm"

    classes = ("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",)

    if len(sys.argv) > 1:
        iname = sys.argv[1]

    c, test_image = readpgm(iname)

    print(f"Loaded image: {iname}")
    show(test_image, ANSI_COLOURS)

    results = inference(test_image)
 
    print(f"Expected: {c}\nPredicted: {classes[np.argmax(results['output'])]}")

if __name__ == "__main__":
    main()
