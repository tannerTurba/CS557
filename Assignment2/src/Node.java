import java.util.*;

public class Node implements Comparable<Node> {
    private ArrayList<Point> data = null;
    private ArrayList<Map<Character, Integer>> attributes = new ArrayList<>();
    private Map<Character, Integer> outputClasses = new HashMap<>();
    private Map<Character, Node> directory = new HashMap<>();
    private Attribute[] allAttributes;
    private Attribute allOutputs;
    private int attrIndex = -1;
    private int verbosity;
    private double infoGain;
    private int depth;
    StringBuilder sb = new StringBuilder();
    private static int nodeNum = 0;
    private static PriorityQueue<Node> frontier = new PriorityQueue<>();

    public Node(ArrayList<Point> data, Attribute[] allAttributes, Attribute allOutputs, int verbosity, int depth) {
        this.data = data;
        this.allAttributes = allAttributes;
        this.allOutputs = allOutputs;
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

    public Node(ArrayList<Point> data, Attribute[] allAttributes, Attribute allOutputs, ArrayList<Map<Character, Integer>> containedAttributes, int verbosity, int depth) {
        this.data = data;
        this.allAttributes = allAttributes;
        this.attributes = containedAttributes;
        this.allOutputs = allOutputs;
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

    private String examine(int depthLimit) {
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
    
        attrIndex = importance(sb);
        return sb.toString();
    }

    public String split(int depthLimit, boolean splitIsLimited) {
        int j = attrIndex;
        ArrayList<Character> vals = new ArrayList<>(allAttributes[j].getValMap().keySet());
        for (int i = 0; i < vals.size(); i++) {
            // construct exs list
            ArrayList<Point> exs = new ArrayList<>();
            for (Point point : data) {
                if (point.containsInput(j, vals.get(i))) {
                    exs.add(point);
                }
            }

            if (!exs.isEmpty()) {
                ArrayList<Map<Character, Integer>> subset = new ArrayList<>(attributes);
                subset.set(j, null);
                
                Node child = new Node(exs, allAttributes, allOutputs, subset, verbosity, depth + 1);
                directory.put(vals.get(i), child);
                attrIndex = j;

                if (splitIsLimited) {
                    child.importance(null);
                    sb.append(child.examine(depthLimit));
                    if (child.attrIndex > Integer.MIN_VALUE) {
                        frontier.add(child);
                    }
                }
                else {
                    sb.append(child.examine(depthLimit));
                    if (child.attrIndex > Integer.MIN_VALUE) {
                        sb.append(child.split(depthLimit, splitIsLimited));
                    }
                }
            }
        }
        return sb.toString();
    }

    public String learn(int depthLimit, boolean splitIsLimited, int splitLimit) {
        if (splitIsLimited) {
            frontier.add(this);
            sb.append(examine(depthLimit));
            
            for (int i = 0; i < splitLimit && !frontier.isEmpty(); i++) {
                Node n = frontier.poll();
                sb.append(n.split(depthLimit, splitIsLimited));
            }
            return sb.toString();
        }
        else {
            sb.append(examine(depthLimit));
            if (attrIndex == Integer.MIN_VALUE) {
                return sb.toString();
            }

            return split(depthLimit, splitIsLimited);
        }
    }

    private int importance(StringBuilder sb) {
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

    public int getAttrIndex() {
        return attrIndex;
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

    // public int getDepth() {
    //     return depth;
    // }

    // public void setDepth(int d) {
    //     depth = d;
    // }

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
