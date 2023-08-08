using MathNet.Numerics.LinearAlgebra;

namespace RealEstate {
    
    public enum InitializationMethod {
        Random,
        Xavier
    }
    
    /// <summary>
    /// Represents a single neuron in the neural network.
    /// </summary>    
    public class Neuron {
        /// <summary>
        /// A vector of the weights for each connection, coming to the neuron from neurons in the previous layer.
        /// </summary>
        public Vector<double> weights { get; set; }
        
        /// <summary>
        /// A constant by which the weighted sum is increased.
        /// </summary>
        public double bias { get; set; }
        
        /// <summary>
        /// Initializes a new neuron with randomly assigned weights.
        /// </summary>
        /// <param name="inputCount">The number of input connections to the neuron.</param>
        /// <param name="method">The method used to initialize the weights.</param>
        /// <exception cref="System.ArgumentOutOfRangeException">
        /// <paramref name="inputCount"/> is not an integer greater than 0.
        /// </exception>
        public Neuron(int inputCount, InitializationMethod method = InitializationMethod.Random) {
            if (inputCount <= 0) {
                throw new ArgumentOutOfRangeException("Invalid number of input connections to the neuron.");
            }

            var rng = new Random();
            bias = 0;

            if (method == InitializationMethod.Random) {
                double stdDev = Math.Sqrt(2.0 / inputCount);
                weights = Vector<double>.Build.Dense(inputCount, _ => rng.NextDouble() * 2.0 * stdDev - stdDev);
            }
            
            else if (method == InitializationMethod.Xavier) {
                double limit = Math.Sqrt(6.0 / (inputCount + 1));
                weights = Vector<double>.Build.Dense(inputCount, _ => rng.NextDouble() * 2.0 * limit - limit);
            }
        }

        /// <summary>
        /// Calculates the raw and activated output for a given input using the specified activation function.
        /// </summary>
        /// <param name="inputs">The input vector.</param>
        /// <param name="activationFunction">The activation function to apply.</param>
        /// <returns>A tuple containing the raw output and the activated output.</returns>
        public (double rawOutput, double activatedOutput) CalculateOutput(Vector<double> inputs, IActivationFunction activationFunction) {
            double rawOutput = weights * inputs + bias;
            double activatedOutput = activationFunction.Activate(rawOutput);
            return (rawOutput, activatedOutput);
        }

        /// <summary>
        /// Updates the weights of the neuron using the specified gradient and learning rate.
        /// </summary>
        /// <param name="gradient">The gradient used to update the weights.</param>
        /// <param name="learningRate">The learning rate.</param>
        public void UpdateWeights(Vector<double> gradient, double learningRate) {
            weights -= learningRate * gradient;
            bias -= learningRate * gradient.Average();
        }
    }

    /// <summary>
    /// Represents a layer of neurons within the neural network.
    /// </summary>
    public class Layer {
        /// <summary>
        /// Gets the list of neurons in the layer.
        /// </summary>
        public List<Neuron> neurons { get; }
        
        /// <summary>
        /// Gets the activation function used by the neurons in the layer.
        /// </summary>
        public IActivationFunction activationFunction { get; }

        /// <summary>
        /// Initializes a new layer with the specified number of neurons and input connections per neuron.
        /// </summary>
        /// <param name="neuronCount">The number of neurons in the layer.</param>
        /// <param name="inputCountPerNeuron">The number of input connections per neuron.</param>
        /// <param name="activationFunction">The activation function to use.</param>
        public Layer(int neuronCount, int inputCountPerNeuron, IActivationFunction activationFunction) {
            neurons = new List<Neuron>(neuronCount);
            for (int i = 0; i < neuronCount; i++) {
                neurons.Add(new Neuron(inputCountPerNeuron));
            }
            this.activationFunction = activationFunction;
        }

        /// <summary>
        /// Performs forward propagation for the layer, calculating raw and activated outputs for the given inputs.
        /// </summary>
        /// <param name="inputs">The input vector.</param>
        /// <returns>A tuple containing the raw outputs and activated outputs as vectors.</returns>
        public (Vector<double> rawOutputs, Vector<double> activatedOutputs) Forward(Vector<double> inputs) {
            Vector<double> rawOutputs = Vector<double>.Build.Dense(neurons.Count);
            Vector<double> activatedOutputs = Vector<double>.Build.Dense(neurons.Count);
            for (int i = 0; i < neurons.Count; i++) {
                var (rawOutput, activatedOutput) = neurons[i].CalculateOutput(inputs, activationFunction);
                rawOutputs[i] = rawOutput;
                activatedOutputs[i] = activatedOutput;
            }
            return (rawOutputs, activatedOutputs);
        }
    }

    /// <summary>
    /// Represents a feedforward neural network.
    /// </summary>
    public class NeuralNetwork {
        /// <summary>
        /// Gets the layers of the neural network.
        /// </summary>
        public List<Layer> layers { get; }

        /// <summary>
        /// Gets the learning rate of the neural network.
        /// </summary>
        public double learningRate { get; }

        private List<Matrix<double>> layersRawOutput;
        private List<Matrix<double>> layersOutput;

        /// <summary>
        /// Initializes a new neural network with the specified input size, hidden layers, output size, and learning rate.
        /// </summary>
        /// <param name="inputCount">The number of input features.</param>
        /// <param name="hiddenLayers">An array defining the number of neurons in each hidden layer.</param>
        /// <param name="outputCount">The number of neurons in the output layer.</param>
        /// <param name="learningRate">The learning rate for training.</param>
        public NeuralNetwork(int inputCount, int[] hiddenLayers, int outputCount, double learningRate) {
            layers = new List<Layer>();
            var previousLayerNeuronCount = inputCount;
            foreach (var neuronCount in hiddenLayers) {
                layers.Add(new Layer(neuronCount, previousLayerNeuronCount, new ReLUFunction()));
                previousLayerNeuronCount = neuronCount;
            }
            layers.Add(new Layer(outputCount, previousLayerNeuronCount, new IdentityFunction()));

            this.learningRate = learningRate;
        }

        /// <summary>
        /// Performs forward propagation through the network for the given inputs.
        /// </summary>
        /// <param name="inputs">The input matrix.</param>
        /// <returns>The softmax output matrix after forward propagation.</returns>
        public Matrix<double> ForwardPropagation(Matrix<double> inputs) {
            layersRawOutput = new List<Matrix<double>>();
            layersOutput = new List<Matrix<double>>();
            
            Matrix<double> currentInputs = inputs;
            foreach (var layer in layers) {
                Matrix<double> rawOutputs = Matrix<double>.Build.Dense(inputs.RowCount, layer.neurons.Count);
                Matrix<double> activatedOutputs = Matrix<double>.Build.Dense(inputs.RowCount, layer.neurons.Count);
                Parallel.For(0, inputs.RowCount, r => {
                    var rowInputs = currentInputs.Row(r);
                    var (rawOutput, activatedOutput) = layer.Forward(rowInputs);
                    rawOutputs.SetRow(r, rawOutput);
                    activatedOutputs.SetRow(r, activatedOutput);
                });
                layersRawOutput.Add(rawOutputs);
                layersOutput.Add(activatedOutputs);
                currentInputs = activatedOutputs;
            }

            Matrix<double> lastLayerOutput = CalculateSoftmaxOutput(currentInputs);
            layersOutput.Add(lastLayerOutput);
            return lastLayerOutput;
        }

        /// <summary>
        /// Performs backpropagation to update the weights of the network based on the error between the target and predicted outputs.
        /// </summary>
        /// <param name="targetMatrix">The target matrix containing true values.</param>
        /// <param name="inputs">The input matrix containing input features.</param>        
        public void BackPropagation(Matrix<double> targetMatrix, Matrix<double> inputs) {
            int batchSize = targetMatrix.RowCount;

            var gradients = layers.Select(layer => new List<Vector<double>>(layer.neurons.Count)).ToList();
            
            for (int b = 0; b < batchSize; b++) {
                Vector<double> targetVector = targetMatrix.Row(b);
                Vector<double> outputError = layersOutput.Last().Row(b) - targetVector;          
                var errors = new List<Vector<double>>();

                // Calculate the error for each layer.
                for (int i = layers.Count - 1; i >= 0; i--) {
                    var layer = layers[i];
                    Vector<double> error;

                    // The error for the last layer is `target - prediction`.
                    if (i == layers.Count - 1) {
                        error = outputError;
                    } 
                    // For all other layers, compute the weighted error as `next layer error * weights * derivative(raw input to the current layer)`
                    else {
                        var nextLayer = layers[i + 1];
                        error = Vector<double>.Build.Dense(layer.neurons.Count);
                        for (int j = 0; j < nextLayer.neurons.Count; j++) {
                            var nextNeuronError = errors[0][j];
                            for (int k = 0; k < layer.neurons.Count; k++) {
                                error[k] += nextNeuronError * nextLayer.neurons[j].weights[k];
                            }
                        }
                        error = error.PointwiseMultiply(layersRawOutput[i].Row(b).Map(layer.activationFunction.Derivate));
                    }

                    errors.Insert(0, error);
                }

                // Calculate the gradients for each layer based on the errors.
                for (int i = 0; i < layers.Count; i++) {
                    var layer = layers[i];
                    var previousLayerOutput = i == 0 ? inputs.Row(b) : layersOutput[i - 1].Row(b);
                    var error = errors[i];
                    for (int j = 0; j < layer.neurons.Count; j++) {
                        var neuron = layer.neurons[j];
                        var gradient = previousLayerOutput * error[j];

                        if (b == 0) {
                            gradients[i].Add(gradient);
                        } else {
                            gradients[i][j] += gradient;
                        }
                    }
                }
            }

            // Update the weights and biases for each neuron in each layer.
            for (int i = 0; i < layers.Count; i++) {
                var layer = layers[i];
                for (int j = 0; j < layer.neurons.Count; j++) {
                    var neuron = layer.neurons[j];
                    var gradient = gradients[i][j] / batchSize;
                    neuron.UpdateWeights(gradient, learningRate);
                }
            }
        }

        /// <summary>
        /// Calculates the softmax output for a given matrix, normalizing the values into probability-like numbers.
        /// </summary>
        /// <param name="layerOutputs">The output matrix of a layer.</param>
        /// <returns>The softmax output as a matrix.</returns>
        public Matrix<double> CalculateSoftmaxOutput(Matrix<double> layerOutputs) {
            int rowCount = layerOutputs.RowCount;
            int columnCount = layerOutputs.ColumnCount;
            var result = Matrix<double>.Build.Dense(rowCount, columnCount);
            
            Parallel.For(0, rowCount, i => {
                double max = layerOutputs.Row(i).Maximum();
                var exp = layerOutputs.Row(i).Map(x => Math.Exp(x - max));
                result.SetRow(i, exp / exp.Sum());
            });

            return result;
        }
    }
}
