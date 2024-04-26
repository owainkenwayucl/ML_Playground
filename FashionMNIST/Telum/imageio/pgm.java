package imageio;
import java.io.FileWriter;
import java.io.IOException;

public class pgm {
    private double[][] image_tensor;
    private String classification;
    
    public pgm() {
        // Nothing!
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

        ret_val = new double[width][height];
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

        for (int i = 0, i<width; i++){
            for (int j = 0; j<height; j++){
                this.image_tensor[i][j] = new_data[i][j];
            }
        }
        this.image_tensor = new_data();
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
                for (int i = 0; j < width; j++) {
                    quantised = (int) (this.image_tensor[i][j] * white);
                    imagefile.write(Integer.toString(quantised) + " ");
                }
                image_file.write("\n")
            }

            image_file.close();

        } catch (IOException e) {
            System.out.println("Error writing to file: " + filename);
            e.printStackTrace();
        }
    }
}