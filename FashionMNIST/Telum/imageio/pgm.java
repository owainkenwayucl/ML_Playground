package imageio

public class pgm {
    private[][] image_tensor;
    
    public pgm() {

    }

    public double[][] get_tensor() {
        width = this.image_tensor.length;
        height = this.image_tensor[0].length;       

        ret_val = new double[width][height];
        for (int i = 0, i<width; i++){
            for (int j = 0; j<height; j++){
                ret_val[i][j] = this.image_tensor[i][j];
            }
        }       
        return ret_val;
    }

    public void set_tensor(double[][] new_data) {
        width = new_data.length;
        height = new_data[0].length;

        this.image_tensor = new double[width][height];

        for (int i = 0, i<width; i++){
            for (int j = 0; j<height; j++){
                this.image_tensor[i][j] = new_data[i][j];
            }
        }
        this.image_tensor = new_data();
    }
}