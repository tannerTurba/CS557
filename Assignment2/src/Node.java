import java.util.*;

public class Node {
    private ArrayList<Point> data = null;
    private ArrayList<Map<Character, Integer>> attributes = new ArrayList<>();
    private Map<Character, Integer> outputClasses = new HashMap<>();
    private Map<Character, Node> directory = new HashMap<>();
    private Attribute[] allAttributes;
    private Attribute allOutputs;
    private int attrIndex = -1;

    public Node(ArrayList<Point> data, Attribute[] allAttributes, Attribute allOutputs) {
        this.data = data;
        this.allAttributes = allAttributes;
        this.allOutputs = allOutputs;
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

    public Node(ArrayList<Point> data, Attribute[] allAttributes, Attribute allOutputs, ArrayList<Map<Character, Integer>> containedAttributes) {
        this.data = data;
        this.allAttributes = allAttributes;
        this.attributes = containedAttributes;
        this.allOutputs = allOutputs;

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

    public void split() {
        int attrCount = 0;
        for (Map<Character, Integer> attribute : attributes) {
            if (attribute != null) {
                attrCount++;
            }
        }

        if (outputClasses.size() == 1 || attrCount == 0) {
            return;
        }
    
        int j = importance();
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
                // subset.remove(j);
                subset.set(j, null);
                
                Node child = new Node(exs, allAttributes, allOutputs, subset);
                directory.put(vals.get(i), child);
                attrIndex = j;
                child.split();
            }
        }
    }

    private int importance() {
        double bestGain = -1.0;
        int bestIndex = -1;

        double gain;
        for (int i = 0; i < attributes.size(); i++) {
            if (attributes.get(i) != null) {
                gain = entropy(outputClasses) - remainingEntropy(i);
    
                if (gain > bestGain) {
                    bestGain = gain;
                    bestIndex = i;
                }
            }
        }

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

            remainder += (sv.size() / data.size()) * entropy(classes);
        }
        return remainder;
    }

    private double entropy(Map<Character, Integer> outputClasses) {
        double hS = 0.0;
        // for (Map.Entry<Character, String> k : allOutputs.getValMap().entrySet()) {
        //     int kCount = outputClasses.get(k.getKey());
        //     int setCount = data.size();

        //     double proportion = kCount / (double) setCount;
        //     hS += proportion * (Math.log(proportion) / Math.log(2));
        // }

        for (Map.Entry<Character, Integer> k : outputClasses.entrySet()) {
            int kCount = k.getValue();
            int setCount = data.size();

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

    // TODO: deterministic or random??
    public char getOutput() {
        int count = -1;
        char result = ' ';
        for (Map.Entry<Character, Integer> entry : outputClasses.entrySet()) {
            if (entry.getValue() > count) {
                result = entry.getKey();
            }
        }
        return result;
    }
}
