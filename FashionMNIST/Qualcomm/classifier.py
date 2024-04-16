import qaic
import numpy as np
model = "fashion_MNIST/aic-binary-dir/programqpc.bin"

def inference(image):
    vit_sess = qaic.Session(model_path = model)
    image = image[np.newaxis,np.newaxis,...].astype(np_type)
    imageset = {}
    imageset["image"] = image

    vit_sess.setup() 
    input_shape, input_type = vit_sess.model_input_shape_dict['image']
    output_shape, output_type = vit_sess.model_output_shape_dict['output']
    output = vit_sess.run(input_dict)

    return output



if __name__ == "__main__":
    from imageio import readpgm
    fname = "test.pgm"

    test_image = readpgm(fname)

    results = inference(test_image)

    print(results)
