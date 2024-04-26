import imageio.pgm;

public class test_pgm {
    public static void main(String arg[]) {
        String test_file = arg[0];
        String test_out = arg[1];

        pgm test_image = new pgm();
        pgm.read_image(test_file);
        pgm.show_image();
        pgm.write_image(test_out);
    }
}