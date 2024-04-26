import com.ibm.onnxmlir.OMModel;
import com.ibm.onnxmlir.OMTensor;
import com.ibm.onnxmlir.OMTensorList;

import java.util.ArrayList;
import java.util.Arrays;

import imageio.pgm;

public class classifier {

    public static float[] inference(pgm image) {
        long[] input_shape = {1l,1l,28l,28l};
        long input_size = 28l * 28l;

        // There has to be a neater way!
        ArrayList<OMTensor> images_ = new ArrayList<OMTensor>();
        images_.add(new OMTensor(image.get_tensor(), input_shape));
        OMTensorList images = new OMTensorList(images_.toArray(new OMTensor[0]));

        // Do the inference.
        OMTensor[] output = OMModel.mainGraph(images).getOmtArray();
        float[] ret_val = output[0].getFloatData();

        //float[] ret_val = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        return ret_val;
    }

    public static int max_at(float[] data) {
        int max_loc = 0;

        for (int i = 0; i < data.length; i++) {
            if (data[i] > max_loc) {
                max_loc = i;
            }
        }

        return max_loc;
    }

    public static void main(String[] args) {
        String[] fashion_classes = {"T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};

        if (args.length == 0) {
            System.out.println("java classifier <image file>");
            System.exit(0);
        }

        pgm image = new pgm();
        image.read_image(args[0]);

        System.out.println("Loaded Image: " + args[0]);
        image.show_image();
        float[] results = inference(image);

        String prediction = fashion_classes[max_at(results)];

        System.out.println("Expected: " + image.get_classification());
        System.out.println("Predicted: " + prediction);

    }
}