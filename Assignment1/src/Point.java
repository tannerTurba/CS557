/**
 * Tanner Turba
 * CS 557 - Machine Learning
 * A class used to keep track of point inputs and outputs. Also handles the 
 * the augmentation of input data. 
 */
public class Point {
    private double[] inputs;
    private double[] augmented;
    private double output;
    private int augmentedDegree = -1;
    
    public Point(String[] data) {
        output = Double.parseDouble(data[data.length - 1]);
        
        inputs = new double[data.length - 1];
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = Double.parseDouble(data[i]);
        }
    }

    /**
     * Augment the point for the specified polynomial degree.
     * @param degree the polynomial degree.
     */
    public void augment(int degree) {
        double[] result = new double[(inputs.length * degree) + 1];

        if (this.augmentedDegree == -1) {
            // place 1 at index 0
            result[0] = 1;
    
            // Copy original array, placing start at index 1
            System.arraycopy(inputs, 0, result, 1, inputs.length);
            this.augmentedDegree = 1;
        }
        else {
            // Copy array into extended array
            System.arraycopy(augmented, 0, result, 0, augmented.length);
        }

        // Fill array with inputs, raised to the specified degree
        for (int deg = augmentedDegree + 1; deg <= degree; deg++) {
            for (int i = 0; i < inputs.length; i++) {
                int index = (deg - 1) * inputs.length + i + 1;
                result[index] = Math.pow(inputs[i], deg);
            }
        }

        this.augmentedDegree = degree;
        this.augmented = result;
    }

    /**
     * @return Double[] return the augmented inputs
     */
    public double[] getAugmented() {
        return augmented;
    }

    /**
     * @return Double[] return the inputs
     */
    public double[] getInputs() {
        return inputs;
    }

    /**
     * @return Double return the output
     */
    public double getOutput() {
        return output;
    }
}