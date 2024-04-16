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



if __name__ == "__main__":
    from imageio import readpgm
    fname = "test.pgm"

    test_image = readpgm(fname)

    results = inference(test_image)

    print(results)
