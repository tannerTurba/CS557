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
        // Sorts through command line args
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

    /**
     * Getter for the filename.
     * @return the filename.
     */
    public String getFileName() {
        return fileName;
    }

    /**
     * Performs mini batch gradient descent on a set of datapoints and to the specified degree.
     * @param data the dataset
     * @param degree the degree 
     * @return the weights of a model. 
     */
    private double[] miniBatchGradientDescent(ArrayList<Point> data, int degree) {
        // Augment points
        for (Point point : data) {
            point.augment(degree);
        }
        
        // Init weight
        int numOfAttrs = data.get(0).getAugmented().length;
        ArrayList<double[]> weights = new ArrayList<>();
        weights.add(0, new double[numOfAttrs]);
        
        // Init vars and some printing
        double currentCost = 9999.0;
        double lastCost = 0.0;
        int numberOfBatches = batchSize <= 0 ? 1 : data.size() / batchSize;
        int t = 0;
        int e = 0;
        if (verbosity > 1) {
            sBuilder.append("      * Beginning mini-batch gradient descent\n");
            sBuilder.append(String.format("        (alpha=%f, epochLimit=%d, batchSize=%d)\n", learningRate, epochLimit, batchSize));
        }
        if (verbosity > 2) {
            sBuilder.append(String.format("        Initial model with zero weights   : Cost = %14.9f", calcError(data, weights.get(0)) / numberOfBatches));
            if (verbosity > 3) {
                printModel(weights.get(t), degree, data.get(0).getInputs().length);
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
            if (isRandom) {
                Collections.shuffle(data);
            }

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
            lastCost = currentCost;
            currentCost = calcError(data, weights.get(t-1)) / numberOfBatches;

            // Printing
            if ((verbosity > 2 && e > 0 && e % 1000 == 0) || verbosity > 4) {
                sBuilder.append(String.format("        After %6d epochs ( %5d iter.): Cost = %14.9f", e, t, currentCost));
                if (verbosity > 3) {
                    printModel(weights.get(t), degree, data.get(0).getInputs().length);
                }
                else {
                    sBuilder.append("\n");
                }
            }
        }
        
        // Printing
        if (verbosity > 2) {
            sBuilder.append(String.format("        After %6d epochs ( %5d iter.): Cost = %14.9f", e, t, currentCost));
            if (verbosity > 3) {
                printModel(weights.get(t), degree, data.get(0).getInputs().length);
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
            printModel(weights.get(t), degree, data.get(0).getInputs().length);
        }

        // Return best weight
        return weights.get(t);
    }

    /**
     * Calculates the derivative that will be scaled by the step size. 
     * @param data the full data set
     * @param oldWeight the most recent weight being used to calculate the new weight
     * @param batchIndices the indices of the current batch
     * @param k the current attribute being inspected.
     * @return
     */
    private double firstChunk(ArrayList<Point> data, double[] oldWeight, int[] batchIndices, int k) {
        double sum = 0.0;
        for (int i : batchIndices) {
            Point dPoint = data.get(i);
            sum += (-2 * dPoint.getAugmented()[k]) * innerChunk(dPoint, oldWeight);
        }
        return (sum / batchIndices.length);
    }

    /**
     * Calculates the error in the mini batch gradient descent formula
     * @param point the point used to calculate the error.
     * @param oldWeight the most recent weight being used to calculate the new weight.
     * @return
     */
    private double innerChunk(Point point, double[] oldWeight) {
        double errorSum = 0.0;
        for (int j = 0; j < point.getAugmented().length; j++) {
            errorSum += oldWeight[j] * point.getAugmented()[j];
        }
        return point.getOutput() - errorSum;
    }

    /**
     * Creates batches from the full data set.
     * @param data the full data set.
     * @param numberOfBatches the number of batches to create.
     * @return Sets of indices from each batch, used to index the full data set.
     */
    private ArrayList<int[]> createBatches(ArrayList<Point> data, int numberOfBatches) {
        ArrayList<int[]> batchIndices = new ArrayList<>();
        for (int b = 0; b < numberOfBatches; b++) {
            // Determine array sizes.
            int size = batchSize; 
            if (numberOfBatches == 1) {
                size = data.size();
            }
            else if (data.size() % batchSize != 0 && b == numberOfBatches - 1) {
                size = data.size() % batchSize;
            }

            // Load with indices
            int[] indices = new int[size];
            for (int i = 0; i < indices.length; i++) {
                indices[i] = (b * batchSize) + i;
            }
            batchIndices.add(b, indices);
        }
        return batchIndices;
    }

    /**
     * Retrieves a fold from the specified dataset
     * @param data the full dataset
     * @param fold the numbered fold to retrieve
     * @return
     */
    private ArrayList<Point> getFold(ArrayList<Point> data, int fold) {
        ArrayList<Point> result = new ArrayList<>();
        
        // Loop from back to end so indexes aren't affected when removing elements
        for (int i = data.size() + fold - kFolds; i >= 0; i -= kFolds) {
            result.add(data.remove(i));
        }
        return result;
    }

    /**
     * Performs linear regression on a set of datapoints, which is a labeled set of data.
     * @param dataPoints a list of labeled data to train on.
     */
    public void regression(ArrayList<Point> dataPoints) {
        // set max if not specified
        if (maxPolyDegree == -1) {
            maxPolyDegree = minPolyDegree;
        }
        if (kFolds <= 1) {
            sBuilder.append("\nSkipping cross-validation.\n");
        }
        
        for (int degree = minPolyDegree; degree <= maxPolyDegree; degree++) {
            sBuilder.append("----------------------------------\n");
            sBuilder.append(String.format("* Using model of degree %d\n", degree));
            if (kFolds > 1) {
                double totalValError = 0.0;
                double totalTrainError = 0.0;
                if (isRandom) {
                    Collections.shuffle(dataPoints);
                }
                
                for (int currentFold = 0; currentFold < kFolds; currentFold++) {
                    // Remove data that is in current fold.
                    ArrayList<Point> trainingSet = new ArrayList<>(dataPoints);
                    ArrayList<Point> x = getFold(trainingSet, currentFold);
                    ArrayList<Point> validationSet = new ArrayList<>(x);
                    x.clear();
                    
                    // Copy to array and fit.
                    sBuilder.append(String.format("  * Training on all data except Fold %d (%d examples)\n", currentFold + 1, trainingSet.size()));
                    double[] fittedModel = miniBatchGradientDescent(trainingSet, degree);
                    
                    // Report training error.
                    double trainingError = calcError(trainingSet, fittedModel);
                    totalTrainError += trainingError;
                    
                    // Estimate validation error of fitted model on augmented validationSet.
                    for (Point point : validationSet) {
                        point.augment(degree);
                    }
                    double validationError = calcError(validationSet, fittedModel);
                    totalValError += validationError;
                    
                    sBuilder.append(String.format("  * Training and validation errors:     %.6f     %.6f\n\n", trainingError, validationError));
                }
                // Compute average validation error across the folds
                sBuilder.append(String.format("  * Average errors across the folds:    %.6f     %.6f\n", totalTrainError/kFolds, totalValError/kFolds));
            }
            else {
                sBuilder.append(String.format("  * Training on all data (%d examples):\n", dataPoints.size()));

                // Fit a polynomial of degree d to all data and report training error.
                double[] fittedModel = miniBatchGradientDescent(dataPoints, degree);

                // Output
                if (verbosity > 1) {
                    printModel(fittedModel, degree, dataPoints.get(0).getInputs().length);
                }

                double trainingError = calcError(dataPoints, fittedModel);
                sBuilder.append(String.format("  * Training error:        %f\n\n", trainingError));
            }
        }
        System.out.println(sBuilder.toString());
    }

    /**
     * Calculates the error from a model based on a training set.
     * @param set the training set
     * @param model the model
     * @return
     */
    private double calcError(ArrayList<Point> set, double[] model) {
        double error = 0.0;
        for (int i = 0; i < set.size(); i++) {
            Point p = set.get(i);
            error += Math.pow(p.getOutput() - calcPredicted(model, p.getAugmented()) , 2);
        }
        return error / set.size();
    }

    /**
     * Calculates the predicted output
     * @param hypo the weights of the hypothesis function
     * @param input the inputs from the training set
     * @return
     */
    private double calcPredicted(double[] hypo, double[] input) {
        double res = 0;
        for (int i = 0; i < input.length; i++) {
            res += hypo[i] * input[i];
        }
        return res;
    }

    /**
     * Converts the model to a string for printing to output.
     * @param model the model to display
     * @param degree the highest degree that is used
     * @param attrCount the number of attributes
     */
    private void printModel(double[] model, int degree, int attrCount) {
        sBuilder.append("        Model: Y = ");
        int attr = 0;
        int deg = 0;
        for (int i = 0; i < model.length; i++) {
            sBuilder.append(String.format("%.4f", Math.abs(model[i])));

            if (attr > 0) {
                sBuilder.append(String.format(" X%d", attr));
            }
            if (deg > 1) {
                sBuilder.append(String.format("^%d", deg));
            }

            if (i + 1 < model.length) {
                if (model[i + 1] < 0) {
                    sBuilder.append(" - ");
                }
                else {
                    sBuilder.append(" + ");
                }
            }

            if (attr < attrCount) {
                attr++;
            }
            if (deg < degree) {
                deg++;
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

        // Perform regression
        driver.regression(dataPoints);
    }
}