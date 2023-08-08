using MathNet.Numerics.LinearAlgebra;

namespace RealEstate {
    /// <summary>
    /// Provides methods for user interaction, including configuring, training, evaluating, and predicting property classes.
    /// </summary>
    class UserOptions {
        /// <summary>
        /// Configures the neural network based on user inputs.
        /// </summary>
        /// <param name="inputCount">The size of the input feature vector provided to the neural network.</param>
        /// <param name="outputCount">The number of outputs from the neural network.</param>
        /// <returns>A configured neural network instance.</returns>
        internal static NeuralNetwork ConfigureNeuralNetwork(int inputCount, int outputCount) {
            // User input for the hidden layers configuration.
            Console.Write("\nEnter the number of neurons in each hidden layer (e.g. 20,10,10 for 20 neurons in the first hidden layer, 10 in the second hidden layer, and 10 in the last hidden layer: ");
            int[] hiddenLayers = Array.ConvertAll(Console.ReadLine().Split(','), int.Parse);

            // User input for the learning rate.
            Console.Write("Enter the learning rate: ");
            double learningRate = double.Parse(Console.ReadLine());

            return new NeuralNetwork(inputCount, hiddenLayers, outputCount, learningRate);
        }

        /// <summary>
        /// Trains the given neural network using the specified train data and targets.
        /// </summary>
        /// <param name="neuralNetwork">The neural network to train.</param>
        /// <param name="trainData">The train data.</param>
        /// <param name="trainTargets">The target classes for the train data.</param>
        internal static void TrainNeuralNetwork(NeuralNetwork neuralNetwork, List<Vector<double>>? trainData, List<int?>? trainTargets) {
            int batchSize = 16;

            // Batching the train data.
            var trainBatches = trainData.Zip(trainTargets, (d, t) => (inputs: Vector<double>.Build.DenseOfEnumerable(d), target: t))
                                         .ToList()
                                         .Batch(batchSize);

            // Training the neural network with each batch.
            foreach (var batch in trainBatches) {
                var batchInputs = batch.Select(b => b.inputs).ToList();
                var batchTargets = batch.Select(b => b.target.Value - 1).ToList();
                neuralNetwork.ForwardPropagation(Matrix<double>.Build.DenseOfRowVectors(batchInputs));
                neuralNetwork.BackPropagation(Matrix<double>.Build.Dense(batchTargets.Count, Preprocessing.UniqueClasses, (i, j) => j == batchTargets[i] ? 1.0 : 0.0), Matrix<double>.Build.DenseOfRowVectors(batchInputs));
            }

            Console.WriteLine("\nThe Neural Network is ready!");
        }

        /// <summary>
        /// Evaluates the given neural network using the specified test data and targets.
        /// </summary>
        /// <param name="neuralNetwork">The neural network to evaluate.</param>
        /// <param name="testData">The test data.</param>
        /// <param name="testTargets">The target classes for the test data.</param>
        internal static void EvaluateNeuralNetwork(NeuralNetwork neuralNetwork, List<Vector<double>>? testData, List<int?>? testTargets) {
            int numberOfClasses = Preprocessing.UniqueClasses;
            int[] truePositives = new int[numberOfClasses];
            int[] falsePositives = new int[numberOfClasses];
            int[] falseNegatives = new int[numberOfClasses];

            // Make the prediction and update the number of true positives/ false positives/ false negatives based on the predicted class.
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

            // For each class, calculate the precision, recall and F1-score based on the number of true positive, false positive and false negative results.
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

            // Calculate the average precision, recall and F1-score.
            Console.WriteLine($"Average Precision: {totalPrecision / numberOfClasses}");
            Console.WriteLine($"Average Recall: {totalRecall / numberOfClasses}");
            Console.WriteLine($"Average F1-Score: {totalF1Score / numberOfClasses}");
        }

        /// <summary>
        /// Predicts the class of a property based on user input and the trained neural network.
        /// </summary>
        /// <param name="neuralNetwork">The neural network used for prediction.</param>
        internal static void PredictPropertyClass(NeuralNetwork neuralNetwork) {
            Vector<double> userInput = GetUserPropertyInput();

            var result = neuralNetwork.ForwardPropagation(Matrix<double>.Build.DenseOfRowVectors(new[] { userInput }));
            int predictedClass = Array.IndexOf(result.Row(0).ToArray(), result.Row(0).Maximum()) + 1;

            Console.WriteLine($"Predicted Class: {predictedClass}");
        }

        /// <summary>
        /// Prompts the user to enter property details and returns a processed vector for neural network input.
        /// </summary>
        /// <returns>A vector containing processed user input.</returns>
        public static Vector<double> GetUserPropertyInput() {
            Console.WriteLine("\nPlease, enter the following property details:");

            double bed = GetUserDoubleInput("Number of bedrooms: ");
            double bath = GetUserDoubleInput("Number of bathrooms: ");
            double acre_lot = GetUserDoubleInput("Acre lot (total property/land size in acres): ");
            double zip_code = GetUserDoubleInput("Zip code (Postal code of the area): ");
            double house_size = GetUserDoubleInput("House size (Building area/living space in square feet): ");

            string listOfStates = "The list of available states: PuertoRico, VirginIslands, Massachusetts, Connecticut, NewHampshire, Vermont, \n" +
                "NewJersey, NewYork, SouthCarolina, Tennessee, RhodeIsland, Virginia, Wyoming, Maine, Georgia, Pennsylvania, WestVirginia, Delaware";
            Console.Write($"{listOfStates}\nState: ");
            
            State state;
            while (!Enum.TryParse(Console.ReadLine(), out state)) {
                Console.WriteLine("Invalid state. Please enter a valid state from the list:");
            }
            
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

        /// <summary>
        /// Prompts the user to enter a double value with a given prompt message.
        /// </summary>
        /// <param name="prompt">The message that will be displayed to the user.</param>
        /// <returns>A double value entered by the user.</returns>
        private static double GetUserDoubleInput(string prompt) {
            double value;
            while (true) {
                Console.Write(prompt);
                if (double.TryParse(Console.ReadLine(), out value)) {
                    return value;
                }
                Console.WriteLine("Invalid input. Please enter a numerical value.");
            }
        }
    }
}