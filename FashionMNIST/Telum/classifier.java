import com.ibm.onnxmlir.OMModel;
import com.ibm.onnxmlir.OMTensor;
import com.ibm.onnxmlir.OMTensorList;

import imageio.pgm;

public class classifier {

    public static float[] inference(pgm image) {
        float[] ret_val = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
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