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

    private ArrayList<Point> trainingSet;
    private ArrayList<Point> validationSet;
    private Layer[] network = new Layer[2];
    private Neuron biasNeuron = new Neuron();

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

                default:
                    break;
            }
        }
    }

    public void readFile() {
        ArrayList<Point> dataPoints = new ArrayList<>();
        // Load full data set from file
        try {
            Scanner scanner = new Scanner(new File(fileName));

            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if (line.charAt(0) != '#') {
                    // Not a comment, so continue
                    dataPoints.add(new Point(line));
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
            int splitIndex = (int)Math.ceil((double)dataPoints.size() * 0.8);
            trainingSet = new ArrayList<>(dataPoints.subList(0, splitIndex));
            validationSet = new ArrayList<>(dataPoints.subList(splitIndex, dataPoints.size()));
        }
    }

    private void featureScale(ArrayList<Point> set) {
        for (Point point : set) {
            double min = Double.MAX_VALUE;
            double max = Double.MIN_NORMAL;
            for (double featureVal : point.getAttributes()) {
                if (featureVal < min) {
                    min = featureVal;
                }
                if (featureVal > max) {
                    max = featureVal;
                }
            }
            point.minMaxNormalize(min, max);
        }
    }

    public void scale() {
        featureScale(trainingSet);
    }

    public void initNetwork() {
        network[0] = new Layer(null, trainingSet.get(0).getAttributes().length, initialWeightVal, biasNeuron);
        for (int i = 1; i < layerSizes.length; i++) {
            network[i] = new Layer(network[i - 1], layerSizes[i], initialWeightVal, biasNeuron);
        }
        network[network.length - 1] = new Layer(network[network.length - 2], trainingSet.get(0).getNumOfClasses(), initialWeightVal, biasNeuron);
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

    private void forwardPropagate(Point data, int exampleIndex, int batchSize) {
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
                for (Map.Entry<Neuron, Double> precedingNeuron : j.getPrecedingNeurons()) {
                    inJ += precedingNeuron.getKey().a[exampleIndex] * precedingNeuron.getValue();
                }
                j.in = inJ;
                if (exampleIndex == 0) {
                    j.a = new double[batchSize];
                }
                j.a[exampleIndex] = j.g(inJ);
            }
        }
    }

    private void backPropagate(Point data, int exampleIndex, int batchSize) {
        // Forward Propagation
        forwardPropagate(data, exampleIndex, batchSize);

        // Back Propagation
        Neuron[] outputLayer = network[network.length - 1].neurons;
        for (int i = 0; i < outputLayer.length; i++) {
            Neuron j = outputLayer[i];
            if (exampleIndex == 0) {
                j.delta = new double[batchSize];
            }
            j.delta[exampleIndex] = j.gPrime(j.in) * (-2.0 * ((data.getOutputClassIndex() == i ? 1 : 0) - j.a[exampleIndex]));
        }
        for (int i = network.length - 2; i >= 2; i--) {
            Layer l = network[i];
            for (Neuron j : l.neurons) {
                double aggregate = 0.0;
                for (Map.Entry<Neuron, Double> jPrime : j.getSucceedingNeurons()) {
                    aggregate += jPrime.getKey().delta[exampleIndex] * jPrime.getValue();
                }
                if (exampleIndex == 0) {
                    j.delta = new double[batchSize];
                }
                j.delta[exampleIndex] = j.gPrime(j.in) * aggregate;
            }
        }
    }

    public void neuralNetworkTrain() {
        int e = 0; 
        int t = 0;
        double[] estimatedOutputs = new double[trainingSet.get(0).getNumOfClasses()];
        int numberOfBatches = batchSize <= 0 ? 1 : trainingSet.size() / batchSize;
        while (true) {
            if (e >= epochLimit) {
                return;
            }
            else {
                boolean shouldStop = true && e != 0;
                for (double o : estimatedOutputs) {
                    if (o > 0.01) {
                        shouldStop = false;
                    }
                }
                if (shouldStop) {
                    return;
                }
            }

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
                for (int k = 1; k < network.length; k++) {
                    Layer l = network[k];
                    for (Neuron j : l.neurons) {
                        for (Map.Entry<Neuron, Double> arc : j.getPrecedingNeurons()) {
                            Neuron i = arc.getKey();
                            double summation = 0.0;
                            for (int exampleIndex = 0; exampleIndex < batch.length; exampleIndex++) {
                                summation += j.delta[exampleIndex] * i.a[exampleIndex];
                            }
                            summation = summation / batch.length;

                            double weight = arc.getValue();
                            double newWeight = weight - (learningRate * summation) - (2 * learningRate * lambda * weight);
                            arc.setValue(newWeight);
                        }
                    }
                }
                t++;

                for (int b = 0; b < batch.length; b++) {
                    Point exampleE = trainingSet.get(batch[b]);
                    Layer outputLayer = network[network.length - 1];
                    for (int k = 0; k < outputLayer.neurons.length; k++) {
                        Neuron j = outputLayer.neurons[k];
                        double absError = Math.abs((exampleE.getOutputClassIndex() == k ? 1 : 0) - j.a[b]);
                        estimatedOutputs[k] = absError;
                    }
                }
            }
            e++;
        }
    }

    public void getAccuracy(ArrayList<Point> set) {
        int counter = 0;
        for (Point point : set) {
            forwardPropagate(point, 0, 1);

            double predictedOutput = Double.MIN_VALUE;
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
        System.out.println(counter / (double)set.size());
        System.out.println(counter);
        System.out.println(set.size());
    }

    public void train() {
        featureScale(trainingSet);
        initNetwork();
        neuralNetworkTrain();
        getAccuracy(trainingSet);
    }

    public static void main(String[] args) {
        Driver driver = new Driver(args);
        driver.readFile();
        driver.train();
        int i = 0;
    }
}
