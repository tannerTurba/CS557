import java.util.*;

/**
 * Tanner Turba
 * October 30, 2024
 * CS 557 - Machine Learning
 * 
 * This class represents a data point from the input data file.
 */
public class Point {
    private char[] inputs;
    private char output;

    /**
     * Creates a Point object by using the line from the file and the specified attributes
     * @param line line from file
     * @param attributes attributes expected in line
     */
    public Point(String line, Attribute[] attributes) {
        this.inputs = new char[attributes.length];
        Scanner scanner = new Scanner(line);

        // For each attribute
        for (int i = 0; i < attributes.length; i++) {
            // Read and insert the data into the array
            String value = scanner.next().trim();
            this.inputs[i] = value.charAt(0);
        }

        // Read the output
        output = scanner.next().trim().charAt(0);
        scanner.close();
    }

    /**
     * Gets the input characters
     * @return
     */
    public char[] getInputs() {
        return inputs;
    }

    /**
     * Gets the output character
     * @return
     */
    public char getOutput() {
        return output;
    }

    /**
     * Determines if the datapoint contains the specified input
     * @param index the index to search
     * @param x the expected input
     * @return
     */
    public boolean containsInput(int index, char x) {
        return inputs[index] == x;
    }

    /**
     * Creates a string representation of the point
     */
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (char c : inputs) {
            sb.append(c + " ");
        }
        sb.append(output + "\n");
        return sb.toString();
    }
}
