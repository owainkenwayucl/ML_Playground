from PyRuntime import OMExecutionSession
import json
import numpy
import time

convert_types = {"f32":"float32",
                 "f16":"float16"}

def inference(image_data, model, classes):
    setup_start = time.time()
    session = OMExecutionSession(model)
    input_signature_json = json.loads(session.input_signature())
    signature = input_signature_json[0]
    input_type = signature["type"]

    images = image_data[0][numpy.newaxis,numpy.newaxis,...].astype(convert_types[input_type])
    
    if (len(image_data) > 1):
        for a in range(1, len(image_data)):
            image = image_data[a][numpy.newaxis,numpy.newaxis,...].astype(convert_types[input_type])
            images = numpy.concatenate((images, image), axis=0)
    imageset = []
    imageset.append(images)
    setup_stop = time.time()
    #print(f"Time in setup: {setup_stop - setup_start}")
    output = {}
    inf_start = time.time()
    output["output"] = session.run(imageset)
    inf_stop = time.time()
    #print(f"Time in inference: {inf_stop - inf_start}")

    matched = match_output(output['output'][0], classes)
    timing = {"inference_setup":(setup_stop - setup_start), "inference_calc":(inf_stop - inf_start)}
    return matched, timing

def match_output(probabilities, classes):
    matched = []
    for a in range(len(probabilities)):
        r = classes[numpy.argmax(probabilities[a])]
        matched.append(r)

    return matched

def compare_results(results,labels):
    n = len(results)
    correct = 0
    for a in range(n):
        if results[a] == labels[a]:
            correct +=1

    return 100*(correct/n)