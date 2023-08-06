using MathNet.Numerics.LinearAlgebra;

namespace RealEstate {
    public class Neuron {
        public Vector<double> weights { get; set; }
        public double bias { get; set; }

        public Neuron(int inputCount) {
            var rng = new Random();
            double stdDev = Math.Sqrt(2.0 / inputCount);
            weights = Vector<double>.Build.Dense(inputCount, _ => rng.NextDouble() * 2.0 * stdDev - stdDev);
            bias = 0;
        }

        public (double rawOutput, double activatedOutput) CalculateOutput(Vector<double> inputs, IActivationFunction activationFunction) {
            double rawOutput = weights * inputs + bias;
            double activatedOutput = activationFunction.Activate(rawOutput);
            return (rawOutput, activatedOutput);
        }

        public void UpdateWeights(Vector<double> gradient, double learningRate) {
            weights -= learningRate * gradient;
            bias -= learningRate * gradient.Average();
        }
    }

    public class Layer {
        public List<Neuron> neurons { get; }
        public IActivationFunction activationFunction { get; }

        public Layer(int neuronCount, int inputCountPerNeuron, IActivationFunction activationFunction) {
            neurons = new List<Neuron>(neuronCount);
            for (int i = 0; i < neuronCount; i++) {
                neurons.Add(new Neuron(inputCountPerNeuron));
            }
            this.activationFunction = activationFunction;
        }

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

    public class NeuralNetwork {
        public List<Layer> layers { get; }
        public double learningRate { get; }
        private List<Vector<double>> layersRawOutput;
        private List<Vector<double>> layersOutput;

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

        public Vector<double> ForwardPropagation(Vector<double> inputs) {
            layersRawOutput = new List<Vector<double>>();
            layersOutput = new List<Vector<double>>();
            Vector<double> currentInputs = inputs;
            
            foreach (var layer in layers) {
                var (rawOutputs, activatedOutputs) = layer.Forward(currentInputs);
                layersRawOutput.Add(rawOutputs);
                layersOutput.Add(activatedOutputs);
                currentInputs = activatedOutputs;
            }

            Vector<double> lastLayerOutput = CalculateSoftmaxOutput(currentInputs);
            layersOutput.Add(lastLayerOutput);
            return lastLayerOutput;
        }
        
        public void BackPropagation(Vector<double> targetVector, Vector<double> inputs) {
            Vector<double> outputError = layersOutput.Last() - targetVector;          
            var errors = new List<Vector<double>>();

            for (int i = layers.Count - 1; i >= 0; i--) {
                var layer = layers[i];
                Vector<double> error;

                if (i == layers.Count - 1) {
                    error = outputError;
                } else {
                    var nextLayer = layers[i + 1];
                    error = Vector<double>.Build.Dense(layer.neurons.Count);
                    for (int j = 0; j < nextLayer.neurons.Count; j++) {
                        var nextNeuronError = errors[0][j];
                        for (int k = 0; k < layer.neurons.Count; k++) {
                            error[k] += nextNeuronError * nextLayer.neurons[j].weights[k];
                        }
                    }
                    error = error.PointwiseMultiply(layersRawOutput[i].Map(layer.activationFunction.Derivate));
                }

                errors.Insert(0, error);
            }

            for (int i = 0; i < layers.Count; i++) {
                var layer = layers[i];
                var previousLayerOutput = i == 0 ? inputs : layersOutput[i - 1];
                var error = errors[i];
                for (int j = 0; j < layer.neurons.Count; j++) {
                    var neuron = layer.neurons[j];
                    var gradient = previousLayerOutput * error[j];
                    if (gradient.Any(Double.IsNaN) || gradient.Any(Double.IsInfinity)) {
                        System.Console.WriteLine();
                    } 
                    neuron.UpdateWeights(gradient, learningRate);
                }
            }
        }

        public Vector<double> CalculateSoftmaxOutput(Vector<double> layerOutputs) {
            double max = layerOutputs.Maximum();
            var exp = layerOutputs.Map(x => Math.Exp(x - max));
            return exp / exp.Sum();
        }
    }
}
