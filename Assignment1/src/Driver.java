import java.io.*;
import java.util.*;

class Driver {
    private String fileName = null;
    private int kFolds = 1;
    private int minPolyDegree = 1;
    private int maxPolyDegree = -1;
    private double learningRate = 0.005;
    private int epochLimit = 10000;
    private int batchSize = 0;
    private boolean isRandom = false;
    private int verbosity = 1;

    public Driver(String[] args) {
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-f":
                    fileName = args[++i];
                    break;
                
                case "-k":
                    kFolds = Integer.parseInt(args[++i]);
                    break;

                case "-d":
                    minPolyDegree = Integer.parseInt(args[++i]);
                    break;

                case "-D":
                    maxPolyDegree = Integer.parseInt(args[++i]);
                    break;

                case "-a":
                    learningRate = Double.parseDouble(args[++i]);
                    break;

                case "-e":
                    epochLimit = Integer.parseInt(args[++i]);
                    break;

                case "-m":
                    batchSize = Integer.parseInt(args[++i]);
                    break;

                case "-r":
                    isRandom = true;
                    break;

                case "-v":
                    verbosity = Integer.parseInt(args[++i]);
                    break;

                default:
                    break;
            }
        }
    }

    public String getFileName() {
        return fileName;
    }

    public String toString() {
        StringBuilder sb = new StringBuilder("\n");
        sb.append(String.format("fileName: %s\n", fileName));
        sb.append(String.format("kFolds: %d\n", kFolds));
        sb.append(String.format("minPolyDegree: %d\n", minPolyDegree));
        sb.append(String.format("maxPolyDegree: %d\n", maxPolyDegree));
        sb.append(String.format("learningRate: %f\n", learningRate));
        sb.append(String.format("epochLimit: %d\n", epochLimit));
        sb.append(String.format("batchSize: %d\n", batchSize));
        sb.append(String.format("isRandom: %s\n", isRandom ? "true" : "false"));
        sb.append(String.format("verbosity: %d\n", verbosity));

        return sb.toString();
    }
    
    public static void main(String[] args) {
        ArrayList<Point> dataPoints = new ArrayList<>();

        // Process command-line args
        Driver driver = new Driver(args);
        String fileName = driver.getFileName();
        if (fileName == null) {
            System.err.println("Provide a file path!");
            return;
        }

        // Load full data set from file
        try {
            Scanner scanner = new Scanner(new File(fileName));

            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if (line.charAt(0) != '#') {
                    // Not a comment, so split by spaces and continue
                    String[] vals = line.split(" ");
    
                    if (vals.length > 1) {
                        dataPoints.add(new Point(vals));
                    }
                }
            }
            scanner.close();
        } 
        catch (FileNotFoundException e) {
            System.err.println("No such file or directory: " + driver.fileName);
        }

        if (driver.kFolds > 1) {
            // Split full data set into k folds
        }
        

        // for (Point point : dataPoints) {
        //     System.out.println(point);
        // }
    }
}