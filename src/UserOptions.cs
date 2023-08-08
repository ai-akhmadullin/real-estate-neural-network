using MathNet.Numerics.LinearAlgebra;

namespace RealEstate {
    class UserOptions {
        internal static NeuralNetwork ConfigureNeuralNetwork(int inputCount, int outputCount) {
            Console.Write("\nEnter the number of neurons in each hidden layer (e.g. 20,10,10 for 20 neurons in the first hidden layer, 10 in the second hidden layer, and 10 in the last hidden layer: ");
            int[] hiddenLayers = Array.ConvertAll(Console.ReadLine().Split(','), int.Parse);

            Console.Write("Enter the learning rate: ");
            double learningRate = double.Parse(Console.ReadLine());

            return new NeuralNetwork(inputCount, hiddenLayers, outputCount, learningRate);
        }

        internal static void TrainNeuralNetwork(NeuralNetwork neuralNetwork, List<Vector<double>>? trainData, List<int?>? trainTargets) {
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

            Console.WriteLine("\nThe Neural Network is ready!");
        }

        internal static void EvaluateNeuralNetwork(NeuralNetwork neuralNetwork, List<Vector<double>>? testData, List<int?>? testTargets) {
            int numberOfClasses = Preprocessing.UniqueClasses;
            int[] truePositives = new int[numberOfClasses];
            int[] falsePositives = new int[numberOfClasses];
            int[] falseNegatives = new int[numberOfClasses];

            for (var i = 0; i < testData.Count; i++) {
                var inputs = testData[i];
                var target = testTargets[i];
                var prediction = neuralNetwork.ForwardPropagation(Matrix<double>.Build.DenseOfRowVectors(new[] { inputs })).Row(0);
                int predictedClass = Array.IndexOf(prediction.ToArray(), prediction.Maximum());
                int actualClass = target.Value - 1;

                if (predictedClass == actualClass) {
                    truePositives[predictedClass]++;
                } else {
                    falsePositives[predictedClass]++;
                    falseNegatives[actualClass]++;
                }
            }

            double totalPrecision = 0;
            double totalRecall = 0;
            double totalF1Score = 0;

            Console.WriteLine();
            for (int i = 0; i < numberOfClasses; i++) {
                double precision = (double)truePositives[i] / (truePositives[i] + falsePositives[i]);
                double recall = (double)truePositives[i] / (truePositives[i] + falseNegatives[i]);
                double f1Score = 2 * (precision * recall) / (precision + recall);

                totalPrecision += precision;
                totalRecall += recall;
                totalF1Score += f1Score;

                Console.WriteLine($"Class {i + 1} - Precision: {precision}, Recall: {recall}, F1-Score: {f1Score}");
            }

            Console.WriteLine($"Average Precision: {totalPrecision / numberOfClasses}");
            Console.WriteLine($"Average Recall: {totalRecall / numberOfClasses}");
            Console.WriteLine($"Average F1-Score: {totalF1Score / numberOfClasses}");
        }


        internal static void PredictPropertyClass(NeuralNetwork neuralNetwork) {
            Vector<double> userInput = GetUserPropertyInput();

            var result = neuralNetwork.ForwardPropagation(Matrix<double>.Build.DenseOfRowVectors(new[] { userInput }));
            int predictedClass = Array.IndexOf(result.Row(0).ToArray(), result.Row(0).Maximum()) + 1;

            Console.WriteLine($"Predicted Class: {predictedClass}");
        }

        public static Vector<double> GetUserPropertyInput() {
            Console.WriteLine("\nPlease, enter the following property details:");

            Console.Write("Number of bedrooms: ");
            double bed = double.Parse(Console.ReadLine());

            Console.Write("Number of bathrooms: ");
            double bath = double.Parse(Console.ReadLine());

            Console.Write("Acre lot (total property/land size in acres): ");
            double acre_lot = double.Parse(Console.ReadLine());

            Console.Write("Zip code (Postal code of the area): ");
            double zip_code = double.Parse(Console.ReadLine());

            Console.Write("House size (Building area/living space in square feet): ");
            double house_size = double.Parse(Console.ReadLine());

            string listOfStates = "The list of available states: PuertoRico, VirginIslands, Massachusetts, Connecticut, NewHampshire, Vermont, \n" +
                "NewJersey, NewYork, SouthCarolina, Tennessee, RhodeIsland, Virginia, Wyoming, Maine, Georgia, Pennsylvania, WestVirginia, Delaware";
            Console.Write($"{listOfStates}\nState: ");
            
            State state;
            Enum.TryParse(Console.ReadLine(), out state);
            
            var property = new Property {
                bed = bed,
                bath = bath,
                acre_lot = acre_lot,
                zip_code = zip_code,
                house_size = house_size,
                state = state
            };

            var properties = new List<Property> { property };
            var standardizedProperties = Preprocessing.Standardize(properties);
            var standardizedProperty = standardizedProperties.First();

            return Preprocessing.PropertyToVector(standardizedProperty);
        }
    }
}