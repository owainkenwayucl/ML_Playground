from PyRuntime import OMExecutionSession
import json
import numpy

#model = "MedMNIST/medmnist_classifier_pathmnist_30_PL.so"
model = "MedMNIST/medmnist_classifier_resnet18_pathmnist_55_20_32bit.so"

convert_types = {"f32":"float32",
                 "f16":"float16"}

def inference(image):
    session = OMExecutionSession(model)
    input_signature_json = json.loads(session.input_signature())
    signature = input_signature_json[0]
    input_type = signature["type"]

    image = image[numpy.newaxis,numpy.newaxis,...].astype(convert_types[input_type])
    imageset = []
    imageset.append(image)
    output = {}
    output["output"] = session.run(imageset)

    return output

def process_image(filename):
    from PIL import Image

    test_image_png = Image.open(filename)
    test_image_c = numpy.array(test_image_png, dtype=numpy.float32)
    test_image = numpy.moveaxis(test_image_c, 2, 0)

    ti_max = numpy.max(test_image)
    ti_min = numpy.min(test_image)

    print(f"Image max {ti_max}, image min {ti_min}")

    # first put in range 0:1
    ti_range = 255.0

    test_image = numpy.divide(test_image, ti_range)

    # then approximate torchvision wtf transform

    test_image = numpy.subtract(test_image, 0.5)
    test_image = numpy.divide(test_image, 0.5)

    ti_max = numpy.max(test_image)
    ti_min = numpy.min(test_image)

    print(f"Normalised image max {ti_max}, image min {ti_min}")
    
    return numpy.copy(test_image, order='C')

def main():
    import sys
    
    iname = "test.png"

    classes = ('adipose','background','debris','lymphocytes','mucus','smooth muscle','normal colon mucosa','cancer-associated stroma','colorectal adenocarcinoma epithelium')

    if len(sys.argv) > 1:
        iname = sys.argv[1]

    test_image = process_image(iname)

    c = "<not defined>"

    tokens = iname.split("_")
    if len(tokens) > 1:
        c = classes[int(tokens[1])]

    print(f"Loaded image: {iname}")
    #show(test_image, ANSI_COLOURS)

    results = inference(test_image)
 
    print(f"Expected: {c}\nPredicted: {classes[numpy.argmax(results['output'])]}")

if __name__ == "__main__":
    main()
