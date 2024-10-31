import java.util.*;

/**
 * Tanner Turba
 * October 30, 2024
 * CS 557 - Machine Learning
 * 
 * This class represents a Node in the decision tree. This 
 * class is responsible for the splitting of each node to create
 * the tree.
 */
public class Node implements Comparable<Node> {
    private ArrayList<Point> data = null;
    private ArrayList<Map<Character, Integer>> attributes = new ArrayList<>();
    private Map<Character, Integer> outputClasses = new HashMap<>();
    private Map<Character, Node> directory = new HashMap<>();
    private Attribute[] allAttributes;
    private int attrIndex = -1;
    private int verbosity;
    private double infoGain;
    private int depth;
    private StringBuilder sb = new StringBuilder();
    private static int nodeNum = 0;
    private static PriorityQueue<Node> frontier = new PriorityQueue<>();

    /**
     * Creates a tree node
     * @param data the contained data points
     * @param allAttributes all possible attributes
     * @param verbosity the verbosity level for output uses
     * @param depth the depth of the node on the tree
     */
    public Node(ArrayList<Point> data, Attribute[] allAttributes, int verbosity, int depth) {
        this.data = data;
        this.allAttributes = allAttributes;
        this.verbosity = verbosity;
        this.depth = depth;

        // Create a frequency map for each attribute in the data
        for (int i = 0; i < data.get(0).getInputs().length; i++) {
            this.attributes.add(i, new HashMap<>());
        }
        
        // For each data point
        for (Point dataPoint : this.data) {
            // update the input frequency map(s)
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

            // update the output frequency map
            char oClass = dataPoint.getOutput();
            if (!outputClasses.containsKey(oClass)) {
                outputClasses.put(oClass, 1);
            }
            else {
                outputClasses.put(oClass, outputClasses.get(oClass) + 1);
            }
        }
    }

    /**
     * Creates a node with predefined attribute frequency maps
     * @param data the contained data points
     * @param allAttributes all possible attributes
     * @param containedAttributes predefined attribute frequency maps
     * @param verbosity the verbosity level for output uses
     * @param depth the depth of the node on the tree
     */
    public Node(ArrayList<Point> data, Attribute[] allAttributes, ArrayList<Map<Character, Integer>> containedAttributes, int verbosity, int depth) {
        this.data = data;
        this.allAttributes = allAttributes;
        this.attributes = containedAttributes;
        this.verbosity = verbosity;
        this.depth = depth;

        // For each data point
        for (Point dataPoint : this.data) {
            // update the output frequency map
            char oClass = dataPoint.getOutput();
            if (!outputClasses.containsKey(oClass)) {
                outputClasses.put(oClass, 1);
            }
            else {
                outputClasses.put(oClass, outputClasses.get(oClass) + 1);
            }
        }
    }

    /**
     * Examines the node to determine if it should be split.
     * @param depthLimit the depth limit for splitting
     * @return a string for logging purposes
     */
    private String examine(int depthLimit) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("      Examining node %d (depth=%d): ", Node.nodeNum, depth));
        Node.nodeNum++;

        // Count the number of attributes left
        int attrCount = 0;
        for (Map<Character, Integer> attribute : attributes) {
            if (attribute != null) {
                attrCount++;
            }
        }

        // Consider reasons for not splitting
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
    
        // Get the index of the attribute that will provide the most information gain upon splitting
        attrIndex = importance(sb);
        return sb.toString();
    }

    /**
     * Splits the node
     * @param depthLimit the depth limit of the tree
     * @param splitIsLimited indicates if a split limit is used
     * @return a string for logging purposes
     */
    public String split(int depthLimit, boolean splitIsLimited) {
        // get the values of the attribute being split on
        ArrayList<Character> vals = new ArrayList<>(attributes.get(attrIndex).keySet());

        // For each value of the attribute
        for (int i = 0; i < vals.size(); i++) {
            // construct the list of example points from all data points
            ArrayList<Point> exs = new ArrayList<>();
            for (Point point : data) {
                if (point.containsInput(attrIndex, vals.get(i))) {
                    exs.add(point);
                }
            }

            if (!exs.isEmpty()) {
                // Create a subset of attributes but remove the current attribute by setting to null
                ArrayList<Map<Character, Integer>> subset = new ArrayList<>(attributes);
                subset.set(attrIndex, null);
                
                // Create new child node and put in directory
                Node child = new Node(exs, allAttributes, subset, verbosity, depth + 1);
                directory.put(vals.get(i), child);

                if (splitIsLimited) {
                    // Using BFS, so determine info gain and put in frontier
                    child.importance(null);
                    sb.append(child.examine(depthLimit));
                    if (child.attrIndex > Integer.MIN_VALUE) {
                        frontier.add(child);
                    }
                }
                else {
                    // Using DFP, so split immediately 
                    sb.append(child.examine(depthLimit));
                    if (child.attrIndex > Integer.MIN_VALUE) {
                        sb.append(child.split(depthLimit, splitIsLimited));
                    }
                }
            }
        }
        return sb.toString();
    }

    /**
     * Starts the learning process/builds the tree
     * @param depthLimit the depth limit to use
     * @param splitLimit the split limit
     * @return a string for logging purposes
     */
    public String learn(int depthLimit, int splitLimit) {
        boolean splitIsLimited = splitLimit > 0;
        if (splitIsLimited) {
            // Use BFS
            StringBuilder sb = new StringBuilder();
            attrIndex = importance(null);
            frontier.add(this);
            
            // While there are splits remaining and nodes in the frontier
            for (int i = 0; i < splitLimit && !frontier.isEmpty(); i++) {
                // Get node and split
                Node n = frontier.poll();
                sb.append(n.split(depthLimit, splitIsLimited));
            }
            return sb.toString();
        }
        else {
            // Use DFS
            sb.append(examine(depthLimit));
            if (attrIndex == Integer.MIN_VALUE) {
                return sb.toString();
            }
            return split(depthLimit, splitIsLimited);
        }
    }

    /**
     * Calculates the importance of the node, which is the best possible gain
     * @param sb
     * @return the index of the most gainful attribute
     */
    private int importance(StringBuilder sb) {
        double bestGain = -1.0;
        int bestIndex = -1;
        double gain;

        // For each attribute
        for (int i = 0; i < attributes.size(); i++) {
            if (attributes.get(i) != null) {
                // Calculate the information gain 
                gain = entropy(outputClasses) - remainingEntropy(i);

                if (verbosity >= 4 && sb != null) {
                    sb.append(String.format("        Gain=%.4f with split on [%s]\n", gain, allAttributes[i].getName()));
                }

                if (gain > bestGain) {
                    // Update bests
                    bestGain = gain;
                    bestIndex = i;
                }
            }
        }
        infoGain = bestGain;
        return bestIndex;
    }

    /**
     * Calculates the remaining entropy of the Node if split on an attribute
     * @param j the index of the attribute to theoretically split on
     * @return the remaining entropy
     */
    private double remainingEntropy(int j) {
        double remainder = 0.0;
        Map<Character, Integer> vj = attributes.get(j);

        // For each attribute and its frequency
        for (Map.Entry<Character, Integer> entry : vj.entrySet()) {
            char v = entry.getKey();

            // Get the set points that contain the attribute value
            ArrayList<Point> sv = new ArrayList<>();
            for (Point point : data) {
                if (point.containsInput(j, v)) {
                    sv.add(point);
                }
            }

            // Get the frequencies of the output classes
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

            // Aggregate the remainder
            remainder += (sv.size() / (double)data.size()) * entropy(classes);
        }
        return remainder;
    }

    /**
     * Calculates the entropy of the Node
     * @param outputClasses the frequencies of the nodes output classes
     * @return the entropy
     */
    private double entropy(Map<Character, Integer> outputClasses) {
        // Count the number of total outputs
        int setCount = 0;
        for (Map.Entry<Character, Integer> k : outputClasses.entrySet()) {
            int kCount = k.getValue();
            setCount += kCount;
        }
        
        // Calculate entropy
        double hS = 0.0;
        for (Map.Entry<Character, Integer> k : outputClasses.entrySet()) {
            int kCount = k.getValue();

            double proportion = kCount / (double) setCount;
            hS += proportion * (Math.log(proportion) / Math.log(2));
        }
        return hS * -1;
    }

    /**
     * Gets the index of the attribute that the node was split on
     * @return
     */
    public int getAttrIndex() {
        return attrIndex;
    }

    /**
     * Gets the directory, which contains the branches and the associated nodes
     * @return
     */
    public Map<Character, Node> getDirectory() {
        return directory;
    }

    /**
     * Gets the predicted output of the node
     * @return
     */
    public char getOutput() {
        int count = -1;
        char result = ' ';

        // Determine which output class has the greater frequency
        for (Map.Entry<Character, Integer> entry : outputClasses.entrySet()) {
            if (entry.getValue() > count) {
                count = entry.getValue();
                result = entry.getKey();
            }
        }
        return result;
    }

    /**
     * Gets the total number of nodes
     * @return
     */
    public static int getNodeCount() {
        return Node.nodeNum;
    }

    /**
     * Gets the information gain
     * @return
     */
    public double getInfoGain() {
        return infoGain;
    }

    /**
     * Part of the Comparable interface, which is used for the PriorityQueue that is
     * used if the split limit is enforced.
     * @param o
     * @return
     */
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
