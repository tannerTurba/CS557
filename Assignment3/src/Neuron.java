import java.util.*;

public class Neuron {
    Map<Neuron, Double> precedingNeurons = new HashMap<>();
    Map<Neuron, Double> succeedingNeurons = new HashMap<>();
    ActivationFunction activationFunction;
    double in;
    double[] a;
    double[] delta;
    double label = -1;

    public Neuron(Neuron[] precedingNeurons, Neuron biasNeuron, double initialWeight, ActivationFunction function, double label) {
        this.activationFunction = function;
        this.label = label;
        if (initialWeight >= 0) {
            this.precedingNeurons.put(biasNeuron, (Math.random() * initialWeight * 2) - initialWeight);
        }
        else {
            this.precedingNeurons.put(biasNeuron, 0.1);
        }

        if (precedingNeurons != null) {
            for (Neuron i : precedingNeurons) {
                double weight = (Math.random() * initialWeight * 2) - initialWeight;
                if (initialWeight >= 0) {
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
