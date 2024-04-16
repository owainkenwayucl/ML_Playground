import qaic
import numpy as np
model = "fashion_MNIST/aic-binary-dir/programqpc.bin"

def inference(image):
    vit_sess = qaic.Session(model_path = model)

    vit_sess.setup() 
    input_shape, input_type = vit_sess.model_input_shape_dict['input']
    output_shape, output_type = vit_sess.model_output_shape_dict['output']
    image = image[np.newaxis,np.newaxis,...].astype(input_type)
    imageset = {}
    imageset["input"] = image
    output = vit_sess.run(imageset)

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
