import java.util.Arrays;

/*
 * Tanner Turba
 * December 3, 2024
 * CS 557 - Machine Learning
 * 
 * This class holds data information pertaining to a data point from the data set. 
 */
public class Point {
    private double[] attributes;
    private int outputClassIndex = -1;
    private int numOfClasses = -1;
    int index = -1;

    /**
     * Constructs a point object
     * @param line the line of text that represents the data point.
     * @param index the index of the data point for printing purposes.
     */
    public Point(String line, int index) {
        String[] splitLine = line.split("[)] [(]");
        String[] inputs = splitLine[0].replace("(", "").split(" ");

        // Record attribute values.
        attributes = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            attributes[i] = Double.parseDouble(inputs[i]);
        }

        // Record output class index.
        String[] outputs = splitLine[1].replace(")", "").split(" ");
        numOfClasses = outputs.length;
        for (int i = 0; i < outputs.length && outputClassIndex == -1; i++) {
            if (outputs[i].equals("1")) {
                outputClassIndex = i;
            }
        }
        this.index = index;
    }

    /**
     * Gets the attributes.
     * @return
     */
    public double[] getAttributes() {
        return attributes;
    }

    /**
     * Gets the output class index.
     * @return
     */
    public int getOutputClassIndex() {
        return outputClassIndex;
    }

    /**
     * Gets the number of classes.
     * @return
     */
    public int getNumOfClasses() {
        return numOfClasses;
    }

    /**
     * Normalizes an attribute of the data size.
     * @param index the attribute to normalize.
     * @param min minimum value
     * @param max maximum value
     */
    public void minMaxNormalize(int index, double min, double max) {
        if (min == max) {
            min = min * -1;
            max = max * -1;
        }
        attributes[index] =  -1 + 2 * ((attributes[index] - min) / (max - min));
    }

    /**
     * The string representation of the data point.
     */
    public String toString() {
        return Arrays.toString(attributes) + " " + outputClassIndex;
    }
}
