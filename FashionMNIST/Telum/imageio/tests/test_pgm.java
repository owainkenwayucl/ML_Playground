package imageio.tests;
import imageio.pgm;

public class test_pgm {
    public static void main(String arg[]) {
        String test_file = arg[0];
        String test_out = arg[1];

        pgm test_image = new pgm();
        test_image.read_image(test_file);
        System.out.println(test_image.get_classification());
        test_image.show_image();
        test_image.write_image(test_out);
    }
}