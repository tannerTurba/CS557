import java.util.*;

public class Node implements Comparable<Node> {
    private ArrayList<Point> data = null;
    private ArrayList<Map<Character, Integer>> attributes = new ArrayList<>();
    private Map<Character, Integer> outputClasses = new HashMap<>();
    private Map<Character, Node> directory = new HashMap<>();
    public int attrIndex = -1;
    private int verbosity;
    private double infoGain;
    private int depth;
    StringBuilder sb = new StringBuilder();
    private static int nodeNum = 0;
    // private static PriorityQueue<Node> frontier = new PriorityQueue<>();

    public Node(ArrayList<Point> data, int verbosity, int depth) {
        this.data = data;
        this.verbosity = verbosity;
        this.depth = depth;
        for (int i = 0; i < data.get(0).getInputs().length; i++) {
            this.attributes.add(i, new HashMap<>());
        }
        for (Point dataPoint : this.data) {

            char[] inputs = dataPoint.getInputs();
            for (int i = 0; i < inputs.length; i++) {
                Map<Character, Integer> attrCounts = this.attributes.get(i);

                if (!attrCounts.containsKey(inputs[i])) {
                    attrCounts.put(inputs[i], 1);
                }
                else {
                    attrCounts.put(inputs[i], attrCounts.get(inputs[i]) + 1);
                }
            } 

            char oClass = dataPoint.getOutput();
            if (!outputClasses.containsKey(oClass)) {
                outputClasses.put(oClass, 1);
            }
            else {
                outputClasses.put(oClass, outputClasses.get(oClass) + 1);
            }
        }
    }

    public Node(ArrayList<Point> data, ArrayList<Map<Character, Integer>> containedAttributes, int verbosity, int depth) {
        this.data = data;
        this.attributes = containedAttributes;
        this.verbosity = verbosity;
        this.depth = depth;

        for (Point dataPoint : this.data) {
            char oClass = dataPoint.getOutput();
            if (!outputClasses.containsKey(oClass)) {
                outputClasses.put(oClass, 1);
            }
            else {
                outputClasses.put(oClass, outputClasses.get(oClass) + 1);
            }
        }
    }

    public String examine(int depthLimit, Attribute[] allAttributes) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("      Examining node %d (depth=%d): ", Node.nodeNum, depth));
        int attrCount = 0;
        Node.nodeNum++;
        for (Map<Character, Integer> attribute : attributes) {
            if (attribute != null) {
                attrCount++;
            }
        }

        if (outputClasses.size() == 1 || attrCount == 0) {
            sb.append("node is pure\n");
            attrIndex = Integer.MIN_VALUE;
            return sb.toString();
        }
        else if (depth == depthLimit) {
            sb.append("node is at max depth\n");
            attrIndex = Integer.MIN_VALUE;
            return sb.toString();
        }
        else if (attrCount == 0) {
            sb.append("node is out of attributes\n");
            attrIndex = Integer.MIN_VALUE;
            return sb.toString();
        }
        else {
            sb.append("node is splittable\n");
        }
    
        attrIndex = importance(sb, allAttributes);
        return sb.toString();
    }

    public int importance(StringBuilder sb, Attribute[] allAttributes) {
        double bestGain = -1.0;
        int bestIndex = -1;

        double gain;
        for (int i = 0; i < attributes.size(); i++) {
            if (attributes.get(i) != null) {
                gain = entropy(outputClasses) - remainingEntropy(i);

                if (verbosity >= 4 && sb != null) {
                    sb.append(String.format("        Gain=%.4f with split on [%s]\n", gain, allAttributes[i].getName()));
                }
    
                if (gain > bestGain) {
                    bestGain = gain;
                    bestIndex = i;
                }
            }
        }
        infoGain = bestGain;
        return bestIndex;
    }

    private double remainingEntropy(int j) {
        double remainder = 0.0;

        Map<Character, Integer> vj = attributes.get(j);
        for (Map.Entry<Character, Integer> entry : vj.entrySet()) {
            char v = entry.getKey();

            ArrayList<Point> sv = new ArrayList<>();
            for (Point point : data) {
                if (point.containsInput(j, v)) {
                    sv.add(point);
                }
            }

            Map<Character, Integer> classes = new HashMap<>();
            for (Point point : sv) {
                char oClass = point.getOutput();
                if (!classes.containsKey(oClass)) {
                    classes.put(oClass, 1);
                }
                else {
                    classes.put(oClass, classes.get(oClass) + 1);
                }
            }
            remainder += (sv.size() / (double)data.size()) * entropy(classes);
        }
        return remainder;
    }

    private double entropy(Map<Character, Integer> outputClasses) {
        int setCount = 0;
        for (Map.Entry<Character, Integer> k : outputClasses.entrySet()) {
            int kCount = k.getValue();
            setCount += kCount;
        }
        
        double hS = 0.0;
        for (Map.Entry<Character, Integer> k : outputClasses.entrySet()) {
            int kCount = k.getValue();

            double proportion = kCount / (double) setCount;
            hS += proportion * (Math.log(proportion) / Math.log(2));
        }
        return hS * -1;
    }

    public Map<Character, Node> getDirectory() {
        return directory;
    }

    public char getOutput() {
        int count = -1;
        char result = ' ';
        for (Map.Entry<Character, Integer> entry : outputClasses.entrySet()) {
            if (entry.getValue() > count) {
                count = entry.getValue();
                result = entry.getKey();
            }
        }
        return result;
    }

    public static int getNodeCount() {
        return Node.nodeNum;
    }

    public double getInfoGain() {
        return infoGain;
    }

    public ArrayList<Point> getData() {
        return data;
    }

    public ArrayList<Map<Character, Integer>> getAttributes() {
        return attributes;
    }

    public int getDepth() {
        return depth;
    }

    public void addToDirectory(char c, Node child) {
        directory.put(c, child);
    }

    @Override
    public int compareTo(Node o) {
        if (this.infoGain > o.getInfoGain()) {
            return -1;
        }
        else if (this.infoGain < o.getInfoGain()) {
            return 1;
        }
        else {
            return 0;
        }
    }
}
