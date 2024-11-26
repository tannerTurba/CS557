import java.io.*;
import java.util.*;

public class Driver {
    private String fileName = "";
    private int[] layerSizes = new int[0];
    private double learningRate = 0.01;
    private int epochLimit = 1000;
    private int batchSize = 1;
    private double lambda = 0.0;
    private boolean isRandom = false;
    private double initialWeightVal = 0.1;
    private int verbosity = 1;
    private ActivationFunction activationFunction = ActivationFunction.LOGISTIC;

    private ArrayList<Point> trainingSet;
    private ArrayList<Point> validationSet;
    private Layer[] network = new Layer[2];
    private Neuron biasNeuron;
    private StringBuilder sb = new StringBuilder();

    public Driver(String[] args) {
        // Reads command line args
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-f":
                    fileName = args[++i];
                    break;

                case "-h":
                    int numOfLayers = Integer.parseInt(args[++i]);
                    network = new Layer[numOfLayers + 2];
                    layerSizes = new int[numOfLayers];
                    for (int layerIndex = 0; layerIndex < numOfLayers; layerIndex++) {
                        layerSizes[layerIndex] = Integer.parseInt(args[++i]);
                    }
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

                case "-l":
                    lambda = Double.parseDouble(args[++i]);
                    break;

                case "-r":
                    isRandom = true;
                    break;

                case "-w":
                    initialWeightVal = Double.parseDouble(args[++i]);
                    break;

                case "-v":
                    verbosity = Integer.parseInt(args[++i]);
                    break;

                case "-g":
                    activationFunction = ActivationFunction.getFunction(args[++i]);

                default:
                    break;
            }
        }
        biasNeuron = new Neuron(activationFunction, -1);
    }

    public void readFile() {
        ArrayList<Point> dataPoints = new ArrayList<>();
        // Load full data set from file
        try {
            Scanner scanner = new Scanner(new File(fileName));
            int dataIndex = 1;

            sb.append(String.format("* Reading %s\n", fileName));
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if (line.charAt(0) != '#') {
                    // Not a comment, so continue
                    dataPoints.add(new Point(line, dataIndex));
                    dataIndex++;
                }
            }
            scanner.close();
        } 
        catch (FileNotFoundException e) {
            System.err.println("No such file or directory: " + fileName);
        }

        // If there is data, split into training and validation sets (80% for training)
        if (dataPoints.size() > 0) {
            if (isRandom) {
                // Shuffle if randomized
                Collections.shuffle(dataPoints);
            }
            sb.append("* Doing train/validation split\n");
            int splitIndex = (int)Math.ceil((double)dataPoints.size() * 0.8);
            trainingSet = new ArrayList<>(dataPoints.subList(0, splitIndex));
            validationSet = new ArrayList<>(dataPoints.subList(splitIndex, dataPoints.size()));
        }
    }

    private void featureScale(ArrayList<Point> set, boolean shouldPrint) {
        if (shouldPrint && verbosity >= 2) {
            sb.append(String.format("  * min/max values on training set:\n"));
        }

        if (!set.isEmpty()) {
            // Init mins and maxs arrays
            int numOfFeatures = set.get(0).getAttributes().length;
            double[] mins = new double[numOfFeatures];
            double[] maxs = new double[numOfFeatures];
            for (int i = 0; i < numOfFeatures; i++) {
                mins[i] = Double.MAX_VALUE;
                maxs[i] = -Double.MAX_VALUE;
            }

            // Find all mins and maxs per feature
            for (int i = 0; i < set.size(); i++) {
                double[] features = set.get(i).getAttributes();
                for (int k = 0; k < numOfFeatures; k++) {
                    if (features[k] < mins[k]) {
                        mins[k] = features[k];
                    }
                    if (features[k] > maxs[k]) {
                        maxs[k] = features[k];
                    }
                }
            }
            
            // Normalize
            for (int i = 0; i < numOfFeatures; i++) {
                if (shouldPrint && verbosity >= 2) {
                    sb.append(String.format("    Feature %d: %.3f, %.3f\n", i + 1, mins[i], maxs[i]));
                }   
                for (Point point : set) {
                    point.minMaxNormalize(i, mins[i], maxs[i]);
                }
            }
        }
    }

    public void initNetwork() {
        if (verbosity >= 2) {
            sb.append("  * Layer sizes (excluding bias neuron(s)):\n");
            sb.append(String.format("    Layer %3d (hidden): %4d\n", 1, trainingSet.get(0).getAttributes().length));
        }

        network[0] = new Layer(null, trainingSet.get(0).getAttributes().length, initialWeightVal, biasNeuron, activationFunction);
        for (int i = 0; i < layerSizes.length; i++) {
            network[i + 1] = new Layer(network[i], layerSizes[i], initialWeightVal, biasNeuron, activationFunction);

            if (verbosity >= 2) {
                sb.append(String.format("    Layer %3d (hidden): %4d\n", i + 2, layerSizes[i]));
            }
        }
        network[network.length - 1] = new Layer(network[network.length - 2], trainingSet.get(0).getNumOfClasses(), initialWeightVal, biasNeuron, ActivationFunction.LOGISTIC);

        if (verbosity >= 2) {
            sb.append(String.format("    Layer %3d (hidden): %4d\n", network.length, trainingSet.get(0).getNumOfClasses()));
        }
    }

    /**
     * Creates batches from the full data set.
     * @param data the full data set.
     * @param numberOfBatches the number of batches to create.
     * @return Sets of indices from each batch, used to index the full data set.
     */
    private int[][] createBatches(ArrayList<Point> data, int numberOfBatches) {
        // ArrayList<int[]> batchIndices = new ArrayList<>();
        int[][] batchIndices = new int[numberOfBatches][];
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
            batchIndices[b] = indices;
            // batchIndices.add(b, indices);
        }
        return batchIndices;
    }

    private void load(Point data, int exampleIndex, int batchSize) {
        Neuron[] inputLayer = network[0].neurons;
        for (int i = 0; i < inputLayer.length; i++) {
            Neuron j = inputLayer[i];
            double x = data.getAttributes()[i];

            if (exampleIndex == 0) {
                j.a = new double[batchSize];
                biasNeuron.a = new double[batchSize];
                for (int k = 0; k < batchSize; k++) {
                    biasNeuron.a[k] = 1.0;
                }
            }
            j.a[exampleIndex] = x;
        }
        for (int i = 1; i < network.length; i++) {
            Layer l = network[i];
            for (Neuron j : l.neurons) {
                double inJ = 0.0;
                for (Map.Entry<Neuron, Double> precedingNeuron : j.precedingNeurons.entrySet()) {
                    inJ += precedingNeuron.getKey().a[exampleIndex] * precedingNeuron.getValue();
                }
                j.in = inJ;

                if (exampleIndex == 0) {
                    j.a = new double[batchSize];
                }
                double a = j.activationFunction.g(inJ);
                j.a[exampleIndex] = a;
            }
        }
    }

    private void forwardPropagate(Point data, int exampleIndex, int batchSize, boolean isLogging) {
        StringBuilder sbIN = new StringBuilder();
        StringBuilder sbA = new StringBuilder();
        if (verbosity >= 4 && isLogging) {
            sb.append(String.format("    * Forward Propagation on example %d\n", data.index));
            sb.append(String.format("      Layer %d %s:    %4s: %5.3f", 1, "(input) ", "a_j", biasNeuron.a[exampleIndex]));
        }
        Neuron[] inputLayer = network[0].neurons;
        for (int i = 0; i < inputLayer.length; i++) {
            Neuron j = inputLayer[i];
            double x = data.getAttributes()[i];

            if (exampleIndex == 0) {
                j.a = new double[batchSize];
                biasNeuron.a = new double[batchSize];
                for (int k = 0; k < batchSize; k++) {
                    biasNeuron.a[k] = 1.0;
                }
            }
            j.a[exampleIndex] = x;

            if (verbosity >= 4 && isLogging) {
                sb.append(String.format(" %5.3f", x));
            }
        }
        for (int i = 1; i < network.length; i++) {
            if (verbosity >= 4 && isLogging) {
                sbIN = new StringBuilder(String.format("\n      Layer %d %8s:    %4s:", i + 1, i == network.length - 1 ? "(output)" : "(hidden)", "in_j"));
                sbA = new StringBuilder("                            a_j:");
            }
            Layer l = network[i];
            for (Neuron j : l.neurons) {
                double inJ = 0.0;
                for (Map.Entry<Neuron, Double> precedingNeuron : j.precedingNeurons.entrySet()) {
                    inJ += precedingNeuron.getKey().a[exampleIndex] * precedingNeuron.getValue();
                }
                j.in = inJ;
                if (verbosity >= 4 && isLogging) {
                    sbIN.append(String.format(" %5.3f", inJ));
                }

                if (exampleIndex == 0) {
                    j.a = new double[batchSize];
                }
                double a = j.activationFunction.g(inJ);
                j.a[exampleIndex] = a;

                if (verbosity >= 4 && isLogging) {
                    sbA.append(String.format(" %5.3f", a));
                }
            }
            if (verbosity >= 4 && isLogging) {
                sb.append(String.format("%s\n%s", sbIN.toString(), sbA.toString()));
            }
        }

        if (verbosity >= 4 && isLogging) {
            sb.append("\n             example's actual y:");
            for (int i = 0; i < data.getNumOfClasses(); i++) {
                if (i == data.getOutputClassIndex()) {
                    sb.append(" 1.000");
                }
                else {
                    sb.append(" 0.000");
                }
            }
            sb.append("\n");
        }
    }

    private void backPropagate(Point data, int exampleIndex, int batchSize) {
        // Forward Propagation
        forwardPropagate(data, exampleIndex, batchSize, true);

        // Back Propagation
        if (verbosity >= 4) {
            sb.append(String.format("    * Backward Propagation on example %d\n", data.index));
            sb.append(String.format("      Layer %d (output): Delta_j:", network.length));
        }

        Neuron[] outputLayer = network[network.length - 1].neurons;
        for (int i = 0; i < outputLayer.length; i++) {
            Neuron j = outputLayer[i];
            if (exampleIndex == 0) {
                j.delta = new double[batchSize];
            }
            double delta = j.activationFunction.gPrime(j.in) * (-2.0 * ((data.getOutputClassIndex() == i ? 1 : 0) - j.a[exampleIndex]));
            j.delta[exampleIndex] = delta;

            if (verbosity >= 4) {
                sb.append(String.format(" %5.3f", delta));
            }
        }
        for (int i = network.length - 2; i >= 1; i--) {
            if (verbosity >= 4) {
                sb.append(String.format("\n      Layer %d (hidden): Delta_j:", i + 1));
            }

            Layer l = network[i];
            for (Neuron j : l.neurons) {
                double aggregate = 0.0;
                for (Map.Entry<Neuron, Double> jPrime : j.succeedingNeurons.entrySet()) {
                    aggregate += jPrime.getKey().delta[exampleIndex] * jPrime.getValue();
                }

                if (exampleIndex == 0) {
                    j.delta = new double[batchSize];
                }
                double delta = j.activationFunction.gPrime(j.in) * aggregate;
                j.delta[exampleIndex] = delta;

                if (verbosity >= 4) {
                    sb.append(String.format(" %5.3f", delta));
                }
            }
        }
        if (verbosity >= 4) {
            sb.append("\n\n");
        }
    }

    public void neuralNetworkTrain() {
        if (verbosity >= 2) {
            sb.append("  * Beginning mini-batch gradient descent\n");
            sb.append(String.format("    (batchSize=%d, epochLimit=%d, learningRate=%.4f, lambda=%.4f)\n", batchSize, epochLimit, learningRate, lambda));
        }

        int e = 0; 
        int t = 0;
        long startTime = System.currentTimeMillis();
        String stopCondition = "";
        double[] estimatedOutputs = new double[trainingSet.get(0).getNumOfClasses()];
        int numberOfBatches = batchSize <= 0 ? 1 : trainingSet.size() / batchSize;
        while (true) {
            if (verbosity >= 3 && e == 0) {
                double totalWeights = 0.0;
                for (int k = 1; k < network.length; k++) {
                    Layer l = network[k];
                    for (Neuron j : l.neurons) {
                        for (Map.Entry<Neuron, Double> arc : j.precedingNeurons.entrySet()) {
                            totalWeights += arc.getValue();
                        }
                    }
                }

                double loss = calcLoss(trainingSet);
                double accuracy = getAccuracy(trainingSet);
                double cost = loss + (lambda * totalWeights);
                sb.append(String.format("    Initial model with random weights : Cost = %.6f; Loss = %.6f; Acc = %.4f\n", cost, loss, accuracy));
            }
            
            if (e >= epochLimit) {
                stopCondition = "Epoch Limit";
                break;
            }
            else {
                boolean shouldStop = true && e != 0;
                for (double o : estimatedOutputs) {
                    if (o > 0.01) {
                        shouldStop = false;
                    }
                }
                if (shouldStop) {
                    stopCondition = "Absolute Error Satisfied";
                    break;
                }
            }
            
            double totalWeights = 0.0;
            int[][] batchIndices = createBatches(trainingSet, numberOfBatches);
            if (isRandom) {
                Collections.shuffle(trainingSet);
            }

            // For each batch
            for (int[] batch : batchIndices) {
                // For each example in the batch
                for (int i = 0; i < batch.length; i++) {
                    Point exampleE = trainingSet.get(batch[i]);
                    backPropagate(exampleE, i, batch.length);
                }

                // For each edge 
                totalWeights = 0.0;
                for (int k = 1; k < network.length; k++) {
                    Layer l = network[k];
                    for (Neuron j : l.neurons) {
                        for (Map.Entry<Neuron, Double> arc : j.precedingNeurons.entrySet()) {
                            Neuron i = arc.getKey();
                            double summation = 0.0;
                            for (int exampleIndex = 0; exampleIndex < batch.length; exampleIndex++) {
                                summation += j.delta[exampleIndex] * i.a[exampleIndex];

                                if (k == network.length - 1) {
                                    // Record largets absolute error for each training example for stopping condition
                                    Point exampleE = trainingSet.get(batch[exampleIndex]);

                                    for (int classIndex = 0; classIndex < l.neurons.length; classIndex++) {
                                        double absError = Math.abs((exampleE.getOutputClassIndex() == classIndex ? 1 : 0) - j.a[exampleIndex]);
                                        if (absError > estimatedOutputs[classIndex]) {
                                            estimatedOutputs[classIndex] = absError;
                                        }
                                    }
                                }
                            }
                            summation = summation / (double)batch.length;

                            double weight = arc.getValue();
                            double newWeight = weight - (learningRate * summation) - (2 * learningRate * lambda * weight);
                            totalWeights += Math.pow(newWeight, 2);
                            arc.setValue(newWeight);
                            i.succeedingNeurons.put(j, newWeight);
                        }
                    }
                }
                t++;
            }
            e++;

            if (verbosity >= 4) {
                double loss = calcLoss(trainingSet);
                double accuracy = getAccuracy(trainingSet);
                double cost = loss + (lambda * totalWeights);
                sb.append(String.format("    After %6d epochs (%6d iter.): Cost = %.6f; Loss = %.6f; Acc = %.4f\n", e, t, cost, loss, accuracy));
            }
            else if (verbosity >= 3 && e % (epochLimit / 10.0) == 0) {
                double loss = calcLoss(trainingSet);
                double accuracy = getAccuracy(trainingSet);
                double cost = loss + (lambda * totalWeights);
                sb.append(String.format("    After %6d epochs (%6d iter.): Cost = %.6f; Loss = %.6f; Acc = %.4f\n", e, t, cost, loss, accuracy));
            }
        }

        if (verbosity >= 2) {
            float totalTime = System.currentTimeMillis() - startTime;
            sb.append("  * Done with fitting!\n");
            sb.append(String.format("    Training took %.0fms, %d epochs, %d iterations (%.4fms / iteration)\n", totalTime, e, t, totalTime / t));
            sb.append(String.format("    GD Stop condition: %s\n", stopCondition));
        }
    }

    public double getAccuracy(ArrayList<Point> set) {
        int counter = 0;
        for (Point point : set) {
            forwardPropagate(point, 0, 1, false);

            double predictedOutput = -Double.MAX_VALUE;
            int predictedClass = -1;
            Layer outputLayer = network[network.length - 1];
            for (int i = 0; i < outputLayer.neurons.length; i++) {
                Neuron n = outputLayer.neurons[i];
                if (predictedOutput < n.a[0]) {
                    predictedClass = i;
                    predictedOutput = n.a[0];
                }
            }
            
            if (predictedClass == point.getOutputClassIndex()) {
                counter++;
            }
        }
        return counter / (double)set.size();
    }

    private double calcLoss(ArrayList<Point> set) {
        double loss = 0.0;
        for (Point point : set) {
            forwardPropagate(point, 0, 1, false);

            Layer outputLayer = network[network.length - 1];
            for (int i = 0; i < outputLayer.neurons.length; i++) {
                Neuron n = outputLayer.neurons[i];
                loss += Math.pow((point.getOutputClassIndex() == i ? 1 : 0) - n.a[0], 2);
            }
        }
        return loss / set.size();
    }

    public void train() {
        if (verbosity >= 1) {
            sb.append("* Scaling features\n");
            featureScale(trainingSet, true);
            featureScale(validationSet, false);
    
            sb.append("* Building network\n");
            initNetwork();
    
            sb.append(String.format("* Training network (using %d examples)\n", trainingSet.size()));
            neuralNetworkTrain();

            double trainingAcc = getAccuracy(trainingSet);
            double validAcc = getAccuracy(validationSet);
            sb.append("* Evaluating accuracy\n");
            sb.append(String.format("  TrainAcc: %.6f\n", trainingAcc));
            sb.append(String.format("  ValidAcc: %.6f\n", validAcc));
        }

        System.out.println(sb);
    }

    public static void main(String[] args) {
        Driver driver = new Driver(args);
        driver.readFile();
        driver.train();
    }
}
