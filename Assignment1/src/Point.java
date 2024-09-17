public class Point {
    private double[] inputs;
    private double output;
    
    public Point(String[] data) {
        output = Double.parseDouble(data[data.length - 1]);
        
        inputs = new double[data.length - 1];
        for (int i = 0; i < inputs.length; i++) {
            inputs[i] = Double.parseDouble(data[i]);
        }
    }

    /**
     * @return Double[] return the inputs
     */
    public double[] getInputs() {
        return inputs;
    }

    // /**
    //  * @param inputs the inputs to set
    //  */
    // public void setInputs(double[] inputs) {
    //     this.inputs = inputs;
    // }

    /**
     * @return Double return the output
     */
    public double getOutput() {
        return output;
    }

    // /**
    //  * @param output the output to set
    //  */
    // public void setOutput(double output) {
    //     this.output = output;
    // }

    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append("Inputs: [");
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
