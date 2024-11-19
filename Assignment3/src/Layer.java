public class Layer {
    Neuron[] neurons; 

    public Layer(Layer previousLayer, int numOfNeurons, double initialWeight, Neuron biasNeuron) {
        neurons = new Neuron[numOfNeurons];

        for (int i = 0; i < numOfNeurons; i++) {
            if (previousLayer == null) {
                // first layer
                neurons[i] = new Neuron();
            }
            else {
                neurons[i] = new Neuron(previousLayer.neurons, biasNeuron, initialWeight);
            }
        }
    }


}
