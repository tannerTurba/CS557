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
    
    private double[] miniBatchGradientDescent(Point[] data) {
        // double[][] weights = new double[data.length][batchSize];
        int numOfAttrs = data[0].getInputs().length;
        
        ArrayList<double[]> weights = new ArrayList<>();
        weights.add(0, new double[numOfAttrs]);
        
        int numberOfBatches = batchSize <= 0 ? 1 : data.length / batchSize;
        int t = 0;
        int e = 0;
        while (e <= epochLimit) {
            ArrayList<int[]> batchIndices = createBatches(data, numberOfBatches);

            // For each batch
            for (int[] indices : batchIndices) {
                double[] newWeight = new double[numOfAttrs];

                // For each k in {0, 1, 2, ..., p}
                for (int k = 0; k < numOfAttrs; k++) {
                    double[] oldWeight = weights.get(t);
                    newWeight[k] = oldWeight[k] - learningRate * firstChunk(data, oldWeight, indices, k);
                }
                weights.add(t + 1, newWeight);
                t++;
            }
            e++;
        }
        return weights.get(t);
    }

    private double firstChunk(Point[] data, double[] oldWeight, int[] batchIndices, int k) {
        double sum = 0.0;
        for (int i : batchIndices) {
            Point dPoint = data[i];
            sum += (-2 * dPoint.getInputs()[k]) * innerChunk(dPoint, oldWeight, i, k);
        }
        return (1.0 / batchIndices.length) * sum;
    }

    private double innerChunk(Point point, double[] oldWeight, int i, int k) {
        double errorSum = 0.0;
        for (int j = 0; j < point.getInputs().length; j++) {
            errorSum += oldWeight[j] * point.getInputs()[j];
        }
        return point.getOutput() - errorSum;
    }

    private ArrayList<int[]> createBatches(Point[] data, int numberOfBatches) {
        ArrayList<int[]> batchIndices = new ArrayList<>();
        for (int b = 0; b < numberOfBatches; b++) {
            int size = batchSize; 
            if (numberOfBatches == 1) {
                size = data.length;
            }
            else if (data.length % batchSize != 0 && b == numberOfBatches - 1) {
                size = data.length % batchSize;
            }

            int[] indices = new int[size];
            for (int i = 0; i < indices.length; i++) {
                indices[i] = (b * batchSize) + i;
            }
            batchIndices.add(b, indices);
        }
        return batchIndices;
    }

    public void regression(ArrayList<Point> dataPoints) {
        if (maxPolyDegree == -1) {
            maxPolyDegree = minPolyDegree;
        }

        int foldSize = 0;
        if (kFolds > 1) {
            // Split full data set into k folds
            foldSize = dataPoints.size() / kFolds + (dataPoints.size() % kFolds);
        }
        for (int degree = minPolyDegree; degree <= maxPolyDegree; degree++) {
            if (kFolds > 1) {
                for (int currentFold = 0; currentFold < kFolds; currentFold++) {
                    // Remove data that is in current fold.
                    ArrayList<Point> copy = new ArrayList<>(dataPoints);
                    copy.subList(currentFold * foldSize, (currentFold * foldSize + foldSize > copy.size() ? copy.size() : currentFold * foldSize + foldSize)).clear();

                    // Copy to array and fit.
                    Point[] dpArray = new Point[copy.size()];
                    copy.toArray(dpArray);
                    double[] fittedModel = miniBatchGradientDescent(dpArray);

                    System.out.print("[");
                    for (double d : fittedModel) {
                        System.out.print(d+ ", ");
                    }
                    System.out.println("]");

                    // TODO: Estimate validation error of fitted model on CURRENTFOLD.
                }
            }
            else {
                // TODO: Fit a polynomial of degree d to all data and report training error.
                // Copy to array and fit.
                Point[] dpArray = new Point[dataPoints.size()];
                dataPoints.toArray(dpArray);
                double[] fittedModel = miniBatchGradientDescent(dpArray);

            }
        }
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

        // Point[] dpArray = new Point[dataPoints.size()];
        // dataPoints.toArray(dpArray);
        driver.regression(dataPoints);
    }
}