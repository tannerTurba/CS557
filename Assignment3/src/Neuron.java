import java.util.*;

public class Neuron {
    Map<Neuron, Double> precedingNeurons = new HashMap<>();
    Map<Neuron, Double> succeedingNeurons = new HashMap<>();
    ActivationFunction activationFunction;
    double in;
    double sampleA = 0.0;
    double[] a;
    double[] delta;
    double label = -1;

    public Neuron(Neuron[] precedingNeurons, Neuron biasNeuron, double initialWeight, ActivationFunction function, double label) {
        this.activationFunction = function;
        this.label = label;
        double weight = (Math.random() * initialWeight * 2) - initialWeight;
        if (initialWeight < 0) {
            weight = 0.1;
        }
        this.precedingNeurons.put(biasNeuron, weight);

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

    // Bias & input Neuron constructor
    public Neuron(ActivationFunction function, double label) {
        this.activationFunction = function;
        this.label = label;
    }
}
