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

    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append("\nInputs: [");
        for (int i = 0; i < inputs.length; i++) {
            sb.append(inputs[i]);

            if (i+1 < inputs.length) {
                sb.append(", ");
            }
        }
        sb.append(String.format("]\nOutput: %s\n", output));

        return sb.toString();
    }
}
