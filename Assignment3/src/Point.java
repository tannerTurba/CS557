import java.util.Arrays;

public class Point {
    private double[] attributes;
    private int outputClassIndex = -1;

    public Point(String line) {
        String[] splitLine = line.split("[)] [(]");
        String[] inputs = splitLine[0].replace("(", "").split(" ");
        attributes = new double[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            attributes[i] = Double.parseDouble(inputs[i]);
        }

        String[] outputs = splitLine[1].replace(")", "").split(" ");
        for (int i = 0; i < outputs.length && outputClassIndex == -1; i++) {
            if (outputs[i].equals("1")) {
                outputClassIndex = i;
            }
        }
    }

    public String toString() {
        return Arrays.toString(attributes) + " " + outputClassIndex;
    }
}
