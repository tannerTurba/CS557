import java.util.*;

/*
 * Tanner Turba
 * December 3, 2024
 * CS 557 - Machine Learning
 * 
 * This class represents a neuron, which holds values for the in value, a value, a 
 * delta value, and its connections to other neurons.
 */
public class Neuron {
    Map<Neuron, Double> precedingNeurons = new HashMap<>();
    Map<Neuron, Double> succeedingNeurons = new HashMap<>();
    ActivationFunction activationFunction;
    double in;
    double sampleA = 0.0;
    double[] a;
    double[] delta;
    double label = -1;

    /**
     * Creates a neuron.
     * @param precedingNeurons the neurons that send values to this neuron.
     * @param biasNeuron the bias neuron.
     * @param initialWeight the intial weight value.
     * @param function the activation function.
     * @param label a label used for deterministic weight initialization.
     */
    public Neuron(Neuron[] precedingNeurons, Neuron biasNeuron, double initialWeight, ActivationFunction function, double label) {
        this.activationFunction = function;
        this.label = label;

        // initialize weight and put bias neuron in the preceding neuron map.
        double weight = (Math.random() * initialWeight * 2) - initialWeight;
        if (initialWeight < 0) {
            weight = 0.1;
        }
        this.precedingNeurons.put(biasNeuron, weight);

        // init weight and put each preceding neuron in the map.
        if (precedingNeurons != null) {
            for (Neuron i : precedingNeurons) {
                weight = (Math.random() * initialWeight * 2) - initialWeight;
                if (initialWeight < 0) {
                    weight = 1.0 / (i.label * Math.pow(2, this.label - 1));
                }

                this.precedingNeurons.put(i, weight);
                i.succeedingNeurons.put(this, weight);
            }
        }
    }

    /**
     * Bias and input neuron constructor.
     * @param function the activation function
     * @param label a label used for deterministic weight initialization.
     */
    public Neuron(ActivationFunction function, double label) {
        this.activationFunction = function;
        this.label = label;
    }
}
