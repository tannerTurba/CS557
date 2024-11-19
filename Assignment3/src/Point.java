import java.util.Arrays;

public class Point {
    private double[] attributes;
    private int outputClassIndex = -1;
    private int numOfClasses = -1;

    public Point(String line) {
        String[] splitLine = line.split("[)] [(]");
        String[] inputs = splitLine[0].replace("(", "").split(" ");
        attributes = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            attributes[i] = Double.parseDouble(inputs[i]);
        }

        String[] outputs = splitLine[1].replace(")", "").split(" ");
        numOfClasses = outputs.length;
        for (int i = 0; i < outputs.length && outputClassIndex == -1; i++) {
            if (outputs[i].equals("1")) {
                outputClassIndex = i;
            }
        }
    }

    public double[] getAttributes() {
        return attributes;
    }

    public int getOutputClassIndex() {
        return outputClassIndex;
    }

    public int getNumOfClasses() {
        return numOfClasses;
    }

    public void minMaxNormalize(double min, double max) {
        if (min == max) {
            min = min * -1;
            max = max * -1;
        }

        double[] normalized = new double[attributes.length];
        for (int i = 0; i < attributes.length; i++) {
            normalized[i] = -1 + 2 * ((attributes[i] - min) / (max - min));
        }
        attributes = normalized;
    }

    private void squaredError() {

    }

    public String toString() {
        return Arrays.toString(attributes) + " " + outputClassIndex;
    }
}
