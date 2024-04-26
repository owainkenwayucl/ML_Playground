package imageio;
import java.io.FileWriter;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.IOException;

public class pgm {
    private double[][] image_tensor;
    private String classification;
   
    public pgm() {
        // Nothing!
    }

    public static String ansi_colour_factory(double colour) {
        String ret_val = "\033[48;5;";

        int colour_ = (int) (colour * 24.0);
        if (colour_ == 24) {
            colour_ = 231;
        } else {
            colour_ = colour_ + 232;
        }
        ret_val = ret_val + Integer.toString(colour_);

        ret_val = ret_val + "m  \033[m";
        return ret_val;
    }

    public void set_classification(String new_classification) {
        this.classification = new String(new_classification);
    }

    public String get_classification(){
        return new String(this.classification);
    }

    public double[][] get_tensor() {
        int width = this.image_tensor.length;
        int height = this.image_tensor[0].length;       

        double[][] ret_val = new double[width][height];
        for (int i = 0; i<width; i++){
            for (int j = 0; j<height; j++){
                ret_val[i][j] = this.image_tensor[i][j];
            }
        }       
        return ret_val;
    }

    public void set_tensor(double[][] new_data) {
        int width = new_data.length;
        int height = new_data[0].length;

        this.image_tensor = new double[width][height];

        for (int i = 0; i<width; i++){
            for (int j = 0; j<height; j++){
                this.image_tensor[i][j] = new_data[i][j];
            }
        }

    }

    public void write_image(String filename) {
        int width = this.image_tensor.length;
        int height = this.image_tensor[0].length; 
        int white = 255;
        int quantised = 0;

        try {
            FileWriter image_file = new FileWriter(filename);

            // Header
            image_file.write("P2\n");
            image_file.write("# " + this.classification + "\n");
            image_file.write(Integer.toString(width) + " " + Integer.toString(height) + "\n");
            image_file.write(Integer.toString(white) + "\n");
            
            // Image
            for (int j = 0; j < height; j++) {
                String line = "";
                for (int i = 0; i < width; i++) {
                    quantised = (int) (this.image_tensor[i][j] * white);
                    line = line + Integer.toString(quantised) + " ";
                }
                line = line + "\n";
                image_file.write(line);
            }

            image_file.close();

        } catch (IOException e) {
            System.out.println("Error writing to file: " + filename);
            e.printStackTrace();
            System.exit(1);
        }
    }

    public void read_image(String filename) {
        int width;
        int height;
        int white;
        int quantised;
        
        try {
            BufferedReader image_file = new BufferedReader(new FileReader(filename));

            String line;
            String[] line_;
            // Header
            line = image_file.readLine().trim();
            if (line.equals("P2") == false) throw new IOException("Not a PGM file!!: " + line);

            this.classification = image_file.readLine().substring(1).trim();

            line = image_file.readLine().trim();
            line_ = line.split(" ");
            width = Integer.parseInt(line_[0]);
            height = Integer.parseInt(line_[1]);

            white = Integer.parseInt(image_file.readLine().trim());
            
            // Image
            this.image_tensor = new double[width][height];

            for (int j = 0; j < height; j++) {
                line = image_file.readLine().trim();
                line_ = line.split(" ");
                for (int i = 0; i < width; i++) {
                    quantised = Integer.parseInt(line_[i]);
                    this.image_tensor[i][j] = (double)quantised/(double)white;
                }
            }

            image_file.close();


        } catch (IOException e) {
            System.out.println("Error reading from file: " + filename);
            e.printStackTrace();
            System.exit(1);
        }

    }

    public void show_image() {
        int width = this.image_tensor.length;
        int height = this.image_tensor[0].length;       

        for (int j = 0; j < height; j++) {
            String line = "";
            for (int i = 0; i < width; i++) {
                line = line + ansi_colour_factory(this.image_tensor[i][j]);
            }
            System.out.println(line);
        }   
    }
}