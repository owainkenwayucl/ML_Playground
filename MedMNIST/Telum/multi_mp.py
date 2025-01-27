from PyRuntime import OMExecutionSession
import json
import numpy
import time
from multiprocessing import Process, Queue, cpu_count

#model = "MedMNIST/medmnist_classifier_pathmnist_30_PL.so"
model = "MedMNIST/medmnist_classifier_resnet18_pathmnist_55_20_32bit.so"

convert_types = {"f32":"float32",
                 "f16":"float16"}

q = Queue()
magic_block = 1000

def chunks(l, n):
    cl = len(l)//n
    cr = len(l)%n
    if cr > 0:
        cl +=1
    return _chunks(l,cl)

def _chunks(l, n):
    for i in range(0,len(l), n):
        yield l[i:i+n]

def inference(index, image_data):
    setup_start = time.time()
    #print(index)
    session = OMExecutionSession(model)
    input_signature_json = json.loads(session.input_signature())
    signature = input_signature_json[0]
    input_type = signature["type"]
    #print(input_type)
    #print(type(image_data[0]))
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

    q.put([index, output])

def mp_inference(image_data, nproc):
    chunked_image_data = list(chunks(image_data, nproc))
    processes = []
    #print(len(chunked_image_data))

    for a in range(nproc):
        processes.append(Process(target=inference, args=(a, chunked_image_data[a])))
        processes[a].start()	

    outputs = {}
    for a in range(nproc):
        print(f"Gathering output from {a}")
        message = q.get()
        outputs[message[0]] = message[1]

    for a in range(nproc):
        print(f"Joining {a}")
        processes[a].join()

    print(f"Done")
    merged_output = []
    for a in range(nproc):
        for b in outputs[a]["output"]:
            merged_output.append(b)

    merged_output = numpy.concatenate(merged_output)

    #print({"output": [merged_output]})
    return {"output": [merged_output]}


def process_image(filename):
    from PIL import Image

    test_image_png = Image.open(filename)
    test_image_c = numpy.array(test_image_png, dtype=numpy.float32)
    test_image = numpy.moveaxis(test_image_c, 2, 0)

    ti_max = numpy.max(test_image)
    ti_min = numpy.min(test_image)

    #print(f"Image max {ti_max}, image min {ti_min}")

    # first put in range 0:1
    ti_range = 255.0

    test_image = numpy.divide(test_image, ti_range)

    # then approximate torchvision wtf transform

    test_image = numpy.subtract(test_image, 0.5)
    test_image = numpy.divide(test_image, 0.5)

    ti_max = numpy.max(test_image)
    ti_min = numpy.min(test_image)

    #print(f"Normalised image max {ti_max}, image min {ti_min}")
    
    return numpy.copy(test_image, order='C')

def main():
    import sys

    iname = ["test.png"]

    classes = ('adipose','background','debris','lymphocytes','mucus','smooth muscle','normal colon mucosa','cancer-associated stroma','colorectal adenocarcinoma epithelium')

    if len(sys.argv) > 1:
        iname = []
        for a in range(1,len(sys.argv)):
            iname.append(sys.argv[a])



    iname_c = chunk(iname, magic_block)
    correct = 0
    n = len(labels)
    nproc = cpu_count()
    print(f"Detected {nproc} CPU cores.")
    start = time.time()
    results = {}
    results["output"] = []
    for it in range(len(iname_c)):
        labels = []
        images = []
        for a in iname_c[it]:
            image = process_image(a)
            images.append(image)
            c = "<not defined>"

            tokens = a.split("_")
            if len(tokens) > 1:
                c = classes[int(tokens[1])]
            labels.append(c)



        results_ = mp_inference(images, nproc)

        # Deep appenc
        for a in results_["output"]:
            results["output"].append(a)
        


    stop = time.time()
    for a in range(n):
        c = labels[a]
        r = classes[numpy.argmax(results['output'][0][a])]
        if (c == r):
            correct = correct + 1
    #    print(f"Expected: {c}\nPredicted: {r}")

    
    print(f"Percentage correct: {100*(correct/n)}%")
    print(f"Time taken for {n} inferences: {stop - start} seconds")

if __name__ == "__main__":
    main()
