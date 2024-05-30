import qaic
import numpy as np
model_dir = "fashion_MNIST/aic-binary-dir"
model_file = "programqpc.bin"

convert_types = {"f32":"float32",
                 "f16":"float16"}

def run(command):
    import subprocess
    return subprocess.run(command, capture_output=True, encoding='UTF-8')

def compile_model(model="fashion_MNIST/fashion_classifier.onnx", batch_size=1):
    import sys
    import os.path

    command = f"/opt/qti-aic/exec/qaic-exec -aic-hw -aic-hw-version=2.0 -compile-only -aic-num-cores=4 -m={model} -onnx-define-symbol=batch_size,{batch_size} -aic-binary-dir=fashion_MNIST/aic-binary-dir_{batch_size}"

    output_file = f"{model_dir}_{batch_size}/{model_file}"

    if (os.path.exists(output_file)): 
        print(f"The compiled file {output_file} already exists, no need to compile.")

    else:
        print(f"The compiled file {output_file} does not exist, compiling.")
        command_list = command.split()
        compile_output = run(command_list)

def inference(image_data):
    batch_size = len(image_data)
    session = qaic.Session(model_path = f"{model_dir}_{batch_size}/{model_file}")

    session.setup() 
    input_shape, input_type = session.model_input_shape_dict['input']
    output_shape, output_type = session.model_output_shape_dict['output']
    
    images = image_data[0][np.newaxis,np.newaxis,...].astype(input_type)
    
    if (len(image_data) > 1):
        for a in range(1, len(image_data)):
            image = image_data[a][np.newaxis,np.newaxis,...].astype(input_type)
            images = np.concatenate((images, image), axis=0)
    imageset = {}
    imageset["input"] = images
    output = session.run(imageset)

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
    
    compile_model(batch_size=n)

    start = time.time()
    results = inference(images)
    stop = time.time()

    print(results)
    correct = 0

    for a in range(n):
        c = labels[a]
        r = classes[np.argmax(results['output'][a])]
        if (c == r):
            correct = correct + 1
        #print(f"Expected: {c}\nPredicted: {r}")

    
    print(f"Percentage correct: {100*(correct/n)}%")
    print(f"Time taken for {n} inferences: {stop - start} seconds")

if __name__ == "__main__":
    main()
