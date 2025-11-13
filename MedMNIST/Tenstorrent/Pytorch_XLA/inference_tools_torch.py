import torch
import torchvision
import numpy
import time
import torch_xla

def inference(image_data, model, classes, device):
    setup_start = time.time()

    num_classes = len(classes)

    if model[0] == "resnet34":
        base_model = torchvision.models.resnet34(num_classes=num_classes)
    elif model[0] == "resnet50":
        base_model = torchvision.models.resnet50(num_classes=num_classes)
    elif model[0] == "resnet101":          
        base_model = torchvision.models.resnet101(num_classes=num_classes)
    elif model[0]== "resnet152":
        base_model = torchvision.models.resnet152(num_classes=num_classes)
    elif model[0] == "wideresnet50":
        base_model = torchvision.models.wide_resnet50_2(num_classes=num_classes)
    elif model[0] == "wideresnet101":
        base_model = torchvision.models.wide_resnet101_2(num_classes=num_classes)
    elif model[0] == "vgg11":
        base_model = torchvision.models.vgg11(num_classes=num_classes)
    else: 
        base_model = torchvision.models.resnet18(num_classes=num_classes)

    base_model.load_state_dict(torch.load(model[1], weights_only=True))

    torch_xla.runtime.set_device_type("TT")
    device = torch_xla.device(int(device))
    base_model.compile(backend="tt")
    base_model = base_model.to(device)
    base_model.eval()

    imageset = torch.from_numpy(numpy.array(image_data)).to(device)

    setup_stop = time.time()

    inf_start = time.time()

    with torch.no_grad():
        probabilities = base_model(imageset).to("cpu").detach().numpy()
    inf_stop = time.time()

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