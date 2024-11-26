import java.util.*;

public class Neuron {
    private Map<Neuron, Double> precedingNeurons = new HashMap<>();
    private Map<Neuron, Double> succeedingNeurons = new HashMap<>();
    double in;
    double[] a;
    double[] delta;

    public Neuron(Neuron[] precedingNeurons, Neuron biasNeuron, double initialWeight) {
        this.precedingNeurons.put(biasNeuron, (Math.random() * initialWeight * 2) - initialWeight);
        if (precedingNeurons != null) {
            for (Neuron neuron : precedingNeurons) {
                double weight = (Math.random() * initialWeight * 2) - initialWeight;
                this.precedingNeurons.put(neuron, weight);
                neuron.addSucceedingNeuron(this, weight);
            }
        }
    }

    public Neuron() {
        // Bias & input Neuron constructor
    }

    // Activation function
    public double g(double inJ) {
        // Use the standard logistic activation function by default
        return 1.0 / (1.0 + Math.pow(Math.E, -inJ));
    }

    // Derivative of activation function
    public double gPrime(double inJ) {
        return g(inJ) * (1.0 - g(inJ));
    }

    public Set<Map.Entry<Neuron, Double>> getPrecedingNeurons() {
        return precedingNeurons.entrySet();
    }

    public void addSucceedingNeuron(Neuron n, double weight) {
        succeedingNeurons.put(n, weight);
    }

    public Set<Map.Entry<Neuron, Double>> getSucceedingNeuronSet() {
        return succeedingNeurons.entrySet();
    }

    public Map<Neuron, Double> getSucceedingNeurons() {
        return succeedingNeurons;
    }
}
