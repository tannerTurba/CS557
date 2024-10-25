import java.util.*;

public class Point {
    private char[] inputs;
    private char output;

    public Point(String line, Attribute[] attributes) {
        this.inputs = new char[attributes.length];
        Scanner scanner = new Scanner(line);

        for (int i = 0; i < attributes.length; i++) {
            String value = scanner.next().trim();
            this.inputs[i] = value.charAt(0);
        }
        output = scanner.next().trim().charAt(0);
 
        scanner.close();
    }

    public char[] getInputs() {
        return inputs;
    }

    public char getOutput() {
        return output;
    }

    public boolean containsInput(int index, char x) {
        return inputs[index] == x;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (char c : inputs) {
            sb.append(c + " ");
        }
        sb.append(output + "\n");
        return sb.toString();
    }
}
