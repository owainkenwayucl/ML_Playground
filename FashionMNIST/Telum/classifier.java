import imageio.pgm;

public class classifier {

    public static double[] inference(pgm image) {
        return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    }

    public static int max_at(double[] data) {
        max_loc = 0;

        for (int i = 0; i < data.length; i++) {
            if data[i] > max loc {
                max_loc = i;
            }
        }

        return max_loc;
    }

    public static void main(String args) {
        String[] fashion_classes = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"];

        if (args.length == 0) {
            System.out.println("java classifier <image file>");
            System.exit(0);
        }

        pgm image = new pgm();
        image.read_image(args[0]);

        System.out.println("Loaded Image: " + args[0]);
        image.show_image();
        double[] results = inference(image);

        String prediction = fashion_classes[max_at(results)];

        System.out.println("Expected: " + image.classification);
        System.out.println("Predicted: " + prediction);

    }
}