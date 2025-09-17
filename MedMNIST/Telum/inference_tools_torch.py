#from PyRuntime import OMExecutionSession
import json
import torch
import torchvision
import numpy
import time

convert_types = {"f32":"float32",
                 "f16":"float16"}

def inference(image_data, model, classes):
    setup_start = time.time()
    #session = OMExecutionSession(model)
    #input_signature_json = json.loads(session.input_signature())
    #signature = input_signature_json[0]
    input_type = "f32"
    base_model = torchvision.models.resnet152(num_classes=len(classes))
    base_model.load_state_dict(torch.load(model, weights_only=True))
    device = torch.device('nnpa')

    base_model = base_model.to(device)
    base_model.eval()
    #images = image_data[0][numpy.newaxis,numpy.newaxis,...].astype(convert_types[input_type])
    
    #if (len(image_data) > 1):
    #    for a in range(1, len(image_data)):
    #        image = image_data[a][numpy.newaxis,numpy.newaxis,...].astype(convert_types[input_type])
    #        images = numpy.concatenate((images, image), axis=0)
    #imageset = []
    #imageset.append(images)

    imageset = torch.from_numpy(numpy.array(image_data)).to(device)

    setup_stop = time.time()
    #print(f"Time in setup: {setup_stop - setup_start}")
    #output = {}
    inf_start = time.time()
    #output["output"] = session.run(imageset)
    probabilities = base_model(imageset).to("cpu").detach().numpy()
    inf_stop = time.time()
    #print(f"Time in inference: {inf_stop - inf_start}")

    matched = match_output(probabilities, classes)
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