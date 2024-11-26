import java.util.*;

public class Neuron {
    Map<Neuron, Double> precedingNeurons = new HashMap<>();
    Map<Neuron, Double> succeedingNeurons = new HashMap<>();
    ActivationFunction activationFunction;
    double in;
    double[] a;
    double[] delta;

    public Neuron(Neuron[] precedingNeurons, Neuron biasNeuron, double initialWeight, ActivationFunction function) {
        this.activationFunction = function;
        this.precedingNeurons.put(biasNeuron, (Math.random() * initialWeight * 2) - initialWeight);
        if (precedingNeurons != null) {
            for (Neuron neuron : precedingNeurons) {
                double weight = (Math.random() * initialWeight * 2) - initialWeight;
                this.precedingNeurons.put(neuron, weight);
                neuron.succeedingNeurons.put(this, weight);
            }
        }
    }

    // Bias & input Neuron constructor
    public Neuron(ActivationFunction function) {
        this.activationFunction = function;
    }
}
