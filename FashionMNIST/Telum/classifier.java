import com.ibm.onnxmlir.OMModel;
import com.ibm.onnxmlir.OMTensor;
import com.ibm.onnxmlir.OMTensorList;

import java.util.ArrayList;
import java.util.Arrays;

import imageio.pgm;

public class classifier {

    public static float[] inference(pgm image) {
        float[][] image_t = image.get_tensor();
        int width = image_t.length;
        int height = image_t[0].length;
        long[] input_shape = {1l,1l,(long)width,(long)height}; 
        long input_size = (long) width * height;
        float[] flat_image = flatten_image(image_t);

        // There has to be a neater way!
        ArrayList<OMTensor> images_ = new ArrayList<OMTensor>();
        images_.add(new OMTensor(flat_image, input_shape));
        OMTensorList images = new OMTensorList(images_.toArray(new OMTensor[0]));

        // Do the inference.
        OMTensor[] output = OMModel.mainGraph(images).getOmtArray();
        float[] ret_val = output[0].getFloatData();

        return ret_val;
    }

    public static int max_at(float[] data) {
        int max_loc = 0;

        for (int i = 0; i < data.length; i++) {
            if (data[i] > data[max_loc]) {
                max_loc = i;
            }
        }

        return max_loc;
    }

    public static float[] flatten_image(float[][] image) {
        int width = image.length;
        int height = image[0].length;
        int input_size = width * height;

        float[] flat_image = new float[width*height];

        int inc = 0;
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
                flat_image[inc] = image[i][j];
                inc++;
            }
        }        
        return flat_image;
    }

    public static String classify(float[] probabilities) {
        String[] fashion_classes = {"T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};
        String prediction = fashion_classes[max_at(probabilities)];
        return prediction;
    }

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("java classifier <image file>");
            System.exit(0);
        }

        pgm image = new pgm();
        image.read_image(args[0]);

        System.out.println("Loaded Image: " + args[0]);
        image.show_image();
        float[] results = inference(image);

        String prediction = classify(results);

        System.out.println("Expected: " + image.get_classification());
        System.out.println("Predicted: " + prediction);

    }
}