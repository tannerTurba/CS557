public class Layer {
    Neuron[] neurons; 

    public Layer(Layer previousLayer, int numOfNeurons, double initialWeight, Neuron biasNeuron, ActivationFunction function) {
        neurons = new Neuron[numOfNeurons];

        for (int i = 0; i < numOfNeurons; i++) {
            if (previousLayer == null) {
                // first layer
                neurons[i] = new Neuron(function);
            }
            else {
                neurons[i] = new Neuron(previousLayer.neurons, biasNeuron, initialWeight, function);
            }
        }
    }
}
