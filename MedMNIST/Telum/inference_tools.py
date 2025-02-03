from PyRuntime import OMExecutionSession
import json
import numpy

convert_types = {"f32":"float32",
                 "f16":"float16"}

def inference(image_data):
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
    print(f"Time in setup: {setup_stop - setup_start}")
    output = {}
    inf_start = time.time()
    output["output"] = session.run(imageset)
    inf_stop = time.time()
    print(f"Time in inference: {inf_stop - inf_start}")

    return output