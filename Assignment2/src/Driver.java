import java.io.*;
import java.util.*;

public class Driver {
    private String filename = null;
    private int trainingGroupSize = 10;
    private int groupSizeIncrememnt = -1;
    private int groupSizeLimit = -1;
    private int numOfTrials = 1;
    private int depthLimit = -1;
    private boolean isRandomized = false;
    private int verbosity = 1;
    private boolean shouldPrintTree = false;
    private int splitLimit = -1;

    private Attribute[] attributes;
    private Attribute outputClasses;
    private ArrayList<Point> points = new ArrayList<>();
    private ArrayList<Point> trainingSet;
    private ArrayList<Point> validationSet;
    private StringBuilder sb = new StringBuilder();

    private Node root;
    
    public Driver(String[] args) {
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-f": 
                    filename = args[++i];
                    break;

                case "-b": 
                    trainingGroupSize = Integer.parseInt(args[++i]);
                    break;

                case "-i": 
                    groupSizeIncrememnt = Integer.parseInt(args[++i]);
                    break;

                case "-l": 
                    groupSizeLimit = Integer.parseInt(args[++i]);
                    break;

                case "-t": 
                    numOfTrials = Integer.parseInt(args[++i]);
                    break;

                case "-d": 
                    depthLimit = Integer.parseInt(args[++i]);
                    break;

                case "-r": 
                    isRandomized = true;
                    break;

                case "-v": 
                    verbosity = Integer.parseInt(args[++i]);
                    break;

                case "-p": 
                    shouldPrintTree = true;
                    break;

                case "-s": 
                    splitLimit = Integer.parseInt(args[++i]);
                    break;

                default :
                break;
            }
        }

        // Set defaults where unspecified
        if (groupSizeIncrememnt == -1) {
            groupSizeIncrememnt = trainingGroupSize;
        }
        if (groupSizeLimit == -1) {
            groupSizeLimit = trainingGroupSize;
        }

        readFile();
    }

    private void readFile() {
        if (filename == null) {
            System.err.println("Provide a file path!");
            return;
        }

        // Load full data set from file
        try {
            Scanner scanner = new Scanner(new File(filename));
            int numOfAttrs = -1;

            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();

                // Not a comment
                if (!line.equals("") && line.charAt(0) != '#') {
                    if (numOfAttrs < 0 && Character.isDigit(line.charAt(0))) {
                        numOfAttrs = Integer.parseInt(line);
                        attributes = new Attribute[numOfAttrs];

                        // Read attrs
                        for (int i = 0; i < numOfAttrs; i++) {
                            line = scanner.nextLine();
                            attributes[i] = new Attribute(line);
                        }   
                    }
                    else if (line.charAt(0) == ':') {
                        // Read output classes
                        outputClasses = new Attribute("output classes " + line);
                    }
                    else {
                        // Read datapoints
                        points.add(new Point(line, attributes));
                    }
                }
            }
            scanner.close();
        } 
        catch (FileNotFoundException e) {
            System.err.println("No such file or directory: " + filename);
        }
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();

        for (Attribute attribute : attributes) {
            sb.append(String.format("%s", attribute));
        }
        sb.append(String.format("%s\n", outputClasses));

        for (Point point : points) {
            sb.append(String.format("%s", point));
        }

        return sb.toString();
    }

    public void decisionTreeLearn() {
        if (isRandomized) {
            Collections.shuffle(points);
        }
        for (int groupSize = trainingGroupSize; groupSize < points.size() && groupSize <= groupSizeLimit; groupSize += groupSizeIncrememnt) {
            double trainingEst = 0.0;
            double validationEst = 0.0;
            int trainingPts = 0;
            int validationPts = 0;
            
            if (verbosity >= 1) {
                sb.append("----------------------------------\n");
                sb.append(String.format("* Using training groups of size %d\n", groupSize));
            }
            
            for (int trial = 1; trial <= numOfTrials; trial++) {
                double trialTrainingEst = 0.0;
                double trialValidationEst = 0.0;
                int trialTrainingPts = 0;
                int trialValidationPts = 0;

                trainingSet = new ArrayList<>(points.subList(0, groupSize));
                validationSet = new ArrayList<>(points.subList(groupSize, points.size()));
        
                root = new Node(trainingSet, attributes, outputClasses, verbosity, 0);
                String output;
                if (splitLimit > 0) {
                    output = root.learn(depthLimit, true, splitLimit);
                }
                else {
                    output = root.learn(depthLimit, false, -1);
                }
                
                trialTrainingEst = guess(root, trainingSet);
                trialValidationEst = guess(root, validationSet);
                trialTrainingPts = trainingSet.size();
                trialValidationPts = validationSet.size();

                if (verbosity >= 2) {
                    sb.append(String.format("  * Trial %d:\n", trial));
                    if (verbosity >= 3) {
                        sb.append("    * Begining decision tree learning\n");
                        sb.append(output);
                        sb.append(String.format("    * Learned tree has %d nodes.\n", Node.getNodeCount()));
                    }
                    sb.append(String.format("    Training and validation accuracy:%12.6f%12.6f\n\n", trialTrainingEst / trialTrainingPts, trialValidationEst / trialValidationPts));
                }
                trainingEst += trialTrainingEst;
                validationEst += trialValidationEst;
                trainingPts += trialTrainingPts;
                validationPts += trialValidationPts;
            }

            if (verbosity >= 1) {
                sb.append(String.format("  * Average accuracy across %d trials:\n", numOfTrials));
                sb.append(String.format("    Training and validation accuracy:%12.6f%12.6f\n\n", trainingEst / trainingPts, validationEst / validationPts));
            }
        }

        System.out.println(sb);
    }

    public double guess(Node n, ArrayList<Point> set) {
        double correctCount = 0;
        for (Point point : set) {
            char result = decisionTreePredict(n, point);
            if (result == point.getOutput()) {
                correctCount++;
            }
        }
        return correctCount;
    }

    public char decisionTreePredict(Node node, Point x) {
        Map<Character, Node> dir = node.getDirectory();
        if (dir.isEmpty()) {
            return node.getOutput();
        }

        int attrIndex = node.getAttrIndex();
        Node nextNode = dir.get(x.getInputs()[attrIndex]);
        if (nextNode == null) {
            // validation error, return best guess
            return node.getOutput();
        }
        return decisionTreePredict(nextNode, x);
    }
    
    public void printTree() {
        if (shouldPrintTree) {
            StringBuilder sb = new StringBuilder("----------------------------------\n");
            sb.append("* Final decision tree:\n");
            sb.append(printNode(root, 0, root.getAttrIndex()));
            System.out.println(sb);
        }
    }

    private String printNode(Node n, int depth, int attrIndex) {
        StringBuilder sb = new StringBuilder();
        
        if (n.getDirectory().isEmpty()) {
            // leaf node
            String output = outputClasses.getValMap().get(n.getOutput());
            sb.append(String.format("Leaf: Predict [%s]\n", output).indent(depth));
        }
        else {
            // child node
            String attrName = attributes[n.getAttrIndex()].getName();
            sb.append(String.format("Node: Split on [%s]\n", attrName).indent(depth));

            for (Map.Entry<Character, Node> branch : n.getDirectory().entrySet()) {
                String branchName = attributes[n.getAttrIndex()].getValMap().get(branch.getKey());
                sb.append(String.format("Branch [%s]=[%s]\n", attrName, branchName != null ? branchName : branch.getKey()).indent(depth + 2));
                sb.append(printNode(branch.getValue(), depth + 4, n.getAttrIndex()));
            }
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        // Process command-line args
        Driver driver = new Driver(args);
        driver.decisionTreeLearn();
        driver.printTree();
    }
}