import java.io.*;
import java.util.*;

public class Driver {
    private String filename = null;
    private int trainingGroupSize = 10;
    private int groupSizeIncrememnt = -1;
    private int groupSizeLimit = -1;
    private int numOfTrials = 1;
    private int maxDepthLimit = -1;
    private boolean isRandomized = false;
    private int verbosity = 1;
    private boolean shouldPrintTree = false;
    private int splitLimit = -1;

    public Attribute[] attributes;
    private Attribute outputClasses;
    private Point[] points;

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
                    maxDepthLimit = Integer.parseInt(args[++i]);
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
            ArrayList<Point> points = new ArrayList<>();
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

            this.points = new Point[points.size()];
            points.toArray(this.points);
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

    public Node decisionTreeLearn() {
        ArrayList<Point> pointList = new ArrayList<>(Arrays.asList(points));
        Node root = new Node(pointList, attributes, outputClasses);
        root.split();
        return root;
    }

    public static void main(String[] args) {
        // Process command-line args
        Driver driver = new Driver(args);
        System.out.println(driver);

        Node tree = driver.decisionTreeLearn();
        int i = 0;
    }
}