/*
 * Tanner Turba
 * December 3, 2024
 * CS 557 - Machine Learning
 * 
 * This is the class that represents a single layer in the neuron network.
 */
public class Layer {
    Neuron[] neurons; 

    /**
     * Initializes the neurons in the layer with the correct connections. 
     * @param previousLayer the previous set of neurons in the network.
     * @param numOfNeurons the number of neurons in the layer.
     * @param initialWeight the initial weight value.
     * @param biasNeuron the bias neuron, which is connected to every other neuron in the network.
     * @param function The activation function.
     */
    public Layer(Layer previousLayer, int numOfNeurons, double initialWeight, Neuron biasNeuron, ActivationFunction function) {
        neurons = new Neuron[numOfNeurons];

        for (int i = 0; i < numOfNeurons; i++) {
            if (previousLayer == null) {
                // first layer
                neurons[i] = new Neuron(function, i + 1);
            }
            else {
                // other layers
                neurons[i] = new Neuron(previousLayer.neurons, biasNeuron, initialWeight, function, i + 1);
            }
        }
    }
}
