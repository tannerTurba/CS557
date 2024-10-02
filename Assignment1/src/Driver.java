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
    private StringBuilder sBuilder = new StringBuilder();

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
    
    private double[] miniBatchGradientDescent(Point[] data, int degree) {
        for (Point point : data) {
            point.augment(degree);
        }
        
        int numOfAttrs = data[0].getAugmented().length;
        ArrayList<double[]> weights = new ArrayList<>();
        weights.add(0, new double[numOfAttrs]);
        
        double currentCost = 9999.0;
        double lastCost = 0.0;
        int numberOfBatches = batchSize <= 0 ? 1 : data.length / batchSize;
        int t = 0;
        int e = 0;
        if (verbosity > 1) {
            sBuilder.append("      * Beginning mini-batch gradient descent\n");
            sBuilder.append(String.format("        (alpha=%f, epochLimit=%d, batchSize=%d)\n", learningRate, epochLimit, batchSize));
        }
        if (verbosity > 2) {
            sBuilder.append(String.format("        Initial model with zero weights   : Cost = %9.9f", calcError(data, weights.get(0)) / numberOfBatches));
            if (verbosity > 3) {
                printModel(weights.get(t));
            }
            else {
                sBuilder.append("\n");
            }
        }
        
        long startTime = System.currentTimeMillis();
        String stopReason = "        GD Stop condition: ";
        while (true) {
            if (e > epochLimit) {
                stopReason += "epochLimit reached\n";
                break;
            }
            else if (currentCost <= Math.pow(10, -10)) {
                stopReason += "CurrentCost ~= 0\n";
                break;
            }
            else if (Math.abs(lastCost - currentCost) <= Math.pow(10, -10)) {
                stopReason += "DeltaCost ~= 0\n";
                break;
            }

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
            lastCost = currentCost;
            currentCost = calcError(data, weights.get(t-1)) / numberOfBatches;
            if (verbosity > 2 && e > 0 && e % 1000 == 0) {
                sBuilder.append(String.format("        After %d epochs ( %d iter.): Cost = %.9f", e, t, currentCost));
                if (verbosity > 3) {
                    printModel(weights.get(t));
                }
                else {
                    sBuilder.append("\n");
                }
            }
            e++;
        }
        
        if (verbosity > 2) {
            sBuilder.append(String.format("        After %d epochs ( %d iter.): Cost = %.9f", e, t, currentCost));
            if (verbosity > 3) {
                printModel(weights.get(t));
            }
            else {
                sBuilder.append("\n");
            }
        }
        if (verbosity > 1) {
            long totalTime = System.currentTimeMillis() - startTime;
            sBuilder.append("      * Done with fitting!\n");
            sBuilder.append(String.format("        Training took %dms, %d epochs, %d iterations (%.4fms / iteration)\n", totalTime, e, t, 9.0/totalTime));
            sBuilder.append(stopReason);
        }

        return weights.get(t);
    }

    private double firstChunk(Point[] data, double[] oldWeight, int[] batchIndices, int k) {
        double sum = 0.0;
        for (int i : batchIndices) {
            Point dPoint = data[i];
            sum += (-2 * dPoint.getAugmented()[k]) * innerChunk(dPoint, oldWeight, i, k);
        }
        return (1.0 / batchIndices.length) * sum;
    }

    private double innerChunk(Point point, double[] oldWeight, int i, int k) {
        double errorSum = 0.0;
        for (int j = 0; j < point.getAugmented().length; j++) {
            errorSum += oldWeight[j] * point.getAugmented()[j];
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
        else {
            sBuilder.append("\nSkipping cross-validation.\n");
        }
        
        for (int degree = minPolyDegree; degree <= maxPolyDegree; degree++) {
            sBuilder.append("----------------------------------\n");
            sBuilder.append(String.format("* Using model of degree %d\n", degree));
            if (kFolds > 1) {
                double[] validationError = new double[kFolds];
                double totalError = 0.0;
                for (int currentFold = 0; currentFold < kFolds; currentFold++) {
                    // Remove data that is in current fold.
                    ArrayList<Point> trainingSet = new ArrayList<>(dataPoints);
                    List<Point> x = trainingSet.subList(currentFold * foldSize, (currentFold * foldSize + foldSize > trainingSet.size() ? trainingSet.size() : currentFold * foldSize + foldSize));
                    ArrayList<Point> validationSet = new ArrayList<>(x);
                    x.clear();

                    // Copy to array and fit.
                    Point[] tSetArray = new Point[trainingSet.size()];
                    trainingSet.toArray(tSetArray);
                    double[] fittedModel = miniBatchGradientDescent(tSetArray, degree);

                    // Report training error.
                    double trainingError = calcError(tSetArray, fittedModel);

                    // Estimate validation error of fitted model on validationSet.
                    Point[] vSetArray = new Point[validationSet.size()];
                    validationSet.toArray(vSetArray);
                    validationError[currentFold] = calcError(vSetArray, fittedModel);
                    totalError += validationError[currentFold];
                }
                // Compute average validation error across the folds
                double avgLoss = totalError / kFolds;
            }
            else {
                sBuilder.append(String.format("  * Training on all data (%d examples):\n", dataPoints.size()));

                // Fit a polynomial of degree d to all data and report training error.
                Point[] dpArray = new Point[dataPoints.size()];
                dataPoints.toArray(dpArray);
                double[] fittedModel = miniBatchGradientDescent(dpArray, degree);

                if (verbosity > 1) {
                    printModel(fittedModel);
                }

                double trainingError = calcError(dpArray, fittedModel);
                sBuilder.append(String.format("  * Training error:        %f\n\n", trainingError));
            }
        }
        System.out.println(sBuilder.toString());
    }

    private double calcError(Point[] set, double[] model) {
        double error = 0.0;
        for (int i = 0; i < set.length; i++) {
            Point p = set[i];
            error += Math.pow(p.getOutput() - calcPredicted(model, p.getAugmented()) , 2);
        }
        return error / set.length;
    }

    private double calcPredicted(double[] hypo, double[] input) {
        double res = 0;
        for (int i = 0; i < input.length; i++) {
            res += hypo[i] * input[i];
        }
        return res;
    }

    private void printModel(double[] model) {
        sBuilder.append("        Model: Y = ");
        for (int i = 0; i < model.length; i++) {
            sBuilder.append(String.format("%.4f", model[i]));
            if (i == 1) {
                sBuilder.append(" X1");
            }
            else if (i > 1) {
                sBuilder.append(String.format(" X1^%d", i));
            }

            if (i + 1 < model.length) {
                sBuilder.append(" + ");
            }
        }
        sBuilder.append("\n");
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