using MathNet.Numerics.LinearAlgebra;

namespace RealEstate {
    class Program {
        public static void Main(string[] args) {
            List<Property> properties = Preprocessing.LoadAndPreprocessData();
            var (trainData, trainTargets, testData, testTargets) = Preprocessing.SplitData(properties, 0.8);

            var inputCount = 6; 
            var hiddenLayers = new int[] { 10 }; 
            var outputCount = Preprocessing.UniqueClasses; 
            var learningRate = 0.01; 
            var neuralNetwork = new NeuralNetwork(inputCount, hiddenLayers, outputCount, learningRate);

            foreach (var (inputs, target) in trainData.Zip(trainTargets, (d, t) => (Vector<double>.Build.DenseOfEnumerable(d), t))) {
                var prediction = neuralNetwork.ForwardPropagation(inputs);
                var targetVector = new double[Preprocessing.UniqueClasses];
                targetVector[target.Value-1] = 1;
                neuralNetwork.BackPropagation(Vector<double>.Build.Dense(targetVector), inputs);
            }

            int correctCount = 0;
            foreach (var (inputs, target) in testData.Zip(testTargets, (d, t) => (Vector<double>.Build.DenseOfEnumerable(d), t))) {
                var prediction = neuralNetwork.ForwardPropagation(inputs);
                int predictedClass = Array.IndexOf(prediction.ToArray(), prediction.Maximum()) + 1;
                if (predictedClass == target) {
                    correctCount++;
                }
            }
            var accuracy = correctCount / (double)testData.Count;
            Console.WriteLine($"Accuracy: {accuracy}");
        }
    }
}
