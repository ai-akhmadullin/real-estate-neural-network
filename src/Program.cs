using MathNet.Numerics.LinearAlgebra;

namespace RealEstate {
    class Program {
        public static void Main(string[] args) {
            List<Property> properties = Preprocessing.LoadAndPreprocessData();
            var (trainData, trainTargets, testData, testTargets) = Preprocessing.SplitData(properties, 0.9);

            var inputCount = trainData[0].Count; 
            var hiddenLayers = new int[] { 10, 10 }; 
            var outputCount = Preprocessing.UniqueClasses; 
            var learningRate = 0.01; 
            var neuralNetwork = new NeuralNetwork(inputCount, hiddenLayers, outputCount, learningRate);

            int batchSize = 16;
            
            var trainBatches = trainData.Zip(trainTargets, (d, t) => (inputs: Vector<double>.Build.DenseOfEnumerable(d), target: t))
                                         .ToList()
                                         .Batch(batchSize);

            foreach (var batch in trainBatches) {
                var batchInputs = batch.Select(b => b.inputs).ToList();
                var batchTargets = batch.Select(b => b.target.Value - 1).ToList();
                neuralNetwork.ForwardPropagation(Matrix<double>.Build.DenseOfRowVectors(batchInputs));
                neuralNetwork.BackPropagation(Matrix<double>.Build.Dense(batchTargets.Count, Preprocessing.UniqueClasses, (i, j) => j == batchTargets[i] ? 1.0 : 0.0), Matrix<double>.Build.DenseOfRowVectors(batchInputs));
            }

            int correctCount = 0;
            foreach (var (inputs, target) in testData.Zip(testTargets, (d, t) => (Vector<double>.Build.DenseOfEnumerable(d), t))) {
                var prediction = neuralNetwork.ForwardPropagation(Matrix<double>.Build.DenseOfRowVectors(new[] { inputs })).Row(0);
                int predictedClass = Array.IndexOf(prediction.ToArray(), prediction.Maximum()) + 1;
                if (predictedClass == target.Value) {
                    correctCount++;
                }
            }

            var accuracy = correctCount / (double)testData.Count;
            Console.WriteLine($"Accuracy: {accuracy}");
        }
    }
}
