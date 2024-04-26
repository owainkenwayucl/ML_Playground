import classifier.fashion_mnist;
import imageio.pgm;

public class classifier {

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("java classifier <image file>");
            System.exit(0);
        }

        pgm image = new pgm();
        image.read_image(args[0]);

        System.out.println("Loaded Image: " + args[0]);
        image.show_image();
        float[] results = fashion_mnist.inference(image);

        String prediction = fashion_mnist.classify(results);

        System.out.println("Expected: " + image.get_classification());
        System.out.println("Predicted: " + prediction);

    }
}