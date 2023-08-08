namespace RealEstate {
    class Program {
        public static void Main(string[] args) {
            Console.WriteLine("Welcome to Real Estate Neural Network Predictor!");

            NeuralNetwork neuralNetwork = null;
            List<Property> properties = Preprocessing.LoadAndPreprocessData();
            var (trainData, trainTargets, testData, testTargets) = Preprocessing.SplitData(properties, 0.9);

            var inputCount = trainData[0].Count; 
            var outputCount = Preprocessing.UniqueClasses; 

            while (true) {
                Console.WriteLine("\nSelect the option:");
                Console.WriteLine("1. Configure Neural Network");
                Console.WriteLine("2. Train Neural Network");
                Console.WriteLine("3. Evaluate Neural Network");
                Console.WriteLine("4. Predict Property Class");
                Console.WriteLine("5. Exit\n");

                switch (Console.ReadLine()) {
                    case "1":
                        neuralNetwork = UserOptions.ConfigureNeuralNetwork(inputCount, outputCount);
                        break;
                    case "2":
                        if (neuralNetwork != null) {
                            UserOptions.TrainNeuralNetwork(neuralNetwork, trainData, trainTargets);
                        } else {
                            Console.WriteLine("Please, configure the neural network first.");
                        }
                        break;
                    case "3":
                        if (neuralNetwork != null) {
                            UserOptions.EvaluateNeuralNetwork(neuralNetwork, testData, testTargets);
                        } else {
                            Console.WriteLine("Please, configure and train the neural network first.");
                        }
                        break;
                    case "4":
                        if (neuralNetwork != null) {
                            UserOptions.PredictPropertyClass(neuralNetwork);
                        } else {
                            Console.WriteLine("Please, configure and train the neural network first.");
                        }
                        break;
                    case "5":
                        return;
                    default:
                        Console.WriteLine("Invalid option. Please, try again.");
                        break;
                }
            }
        }
    }
}