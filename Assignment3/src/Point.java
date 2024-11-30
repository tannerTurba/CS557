import java.util.Arrays;

public class Point {
    private double[] attributes;
    private int outputClassIndex = -1;
    private int numOfClasses = -1;
    int index = -1;

    public Point(String line, int index) {
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
        this.index = index;
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

    public void minMaxNormalize(int index, double min, double max) {
        if (min == max) {
            min = min * -1;
            max = max * -1;
        }
        attributes[index] =  -1 + 2 * ((attributes[index] - min) / (max - min));
    }

    public String toString() {
        return Arrays.toString(attributes) + " " + outputClassIndex;
    }
}
