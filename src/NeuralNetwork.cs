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
        private List<Matrix<double>> layersRawOutput;
        private List<Matrix<double>> layersOutput;

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


        
        public void BackPropagation(Matrix<double> targetMatrix, Matrix<double> inputs) {
            int batchSize = targetMatrix.RowCount;
            
            var gradients = layers.Select(layer => new List<Vector<double>>(layer.neurons.Count)).ToList();
            
            for (int b = 0; b < batchSize; b++) {
                Vector<double> targetVector = targetMatrix.Row(b);
                Vector<double> outputError = layersOutput.Last().Row(b) - targetVector;          
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
                        error = error.PointwiseMultiply(layersRawOutput[i].Row(b).Map(layer.activationFunction.Derivate));
                    }

                    errors.Insert(0, error);
                }

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

            for (int i = 0; i < layers.Count; i++) {
                var layer = layers[i];
                for (int j = 0; j < layer.neurons.Count; j++) {
                    var neuron = layer.neurons[j];
                    var gradient = gradients[i][j] / batchSize;
                    neuron.UpdateWeights(gradient, learningRate);
                }
            }
        }


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
