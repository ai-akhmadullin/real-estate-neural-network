# US Real Estate Prediction System

## Introduction

### Project Overview
The US Real Estate Prediction System is an application designed to predict property classes ranging from low to high based on various features. It uses an artificial neural network and machine learning algorithms to make predictions, offering valuable insights for investors, real estate agents, and individuals interested in the housing market.

### Dataset Description
The system operates on a dataset sourced from Kaggle, containing over 700,000 entries with the following features:

- `status`: Housing status, either ready for sale or ready to be built.
- `bed`: Number of bedrooms.
- `bath`: Number of bathrooms.
- `acre_lot`: Total property/land size in acres.
- `city`: City name.
- `state`: State name.
- `zip_code`: Postal code of the area.
- `house_size`: House area/living space in square feet.
- `prev_sold_date`: Previously sold date.
- `price`: Housing price, either the current listing price or recently sold price if sold recently.

### Data Pre-processing
Before training the neural network, the data undergoes several pre-processing steps:

1. Loading: The data is loaded from a CSV file (realtor-data.csv) into a structured format suitable for processing.
2. Filtering: Since most samples have low-valued features while some samples occur to have unexpected peaks in feature values (can be seen on the graphs below), the data is filtered to keep only those samples whose features are below the 95-th percentile. Apart from it, only those samples are kept which do not possess missing feature values (except for the zip code, which is dealt with later by replacing missing values with an average value).
3. Feature Standardization: Continuous features like house size, acres, bedrooms, etc., are standardized to have a mean of zero and standard deviation of one, which is called z-score normalization.
4. Categorical Conversion: Categorical feature `state` is converted into numerical representation using one-hot encoding.
5. Partitioning: The dataset is partitioned into train and test sets, allowing for both training the model and evaluating its performance on unseen data.

## Main Components

### 1. Program.cs (Main Entry Point)
#### Overview
The main entry point of the system, responsible for initializing and controlling other components. It handles the user interface – commands to train, evaluate, and predict the class of the property using the neural network.

#### Main Functionality
- Initializing the application.
- Managing user interactions and commands.
- Calling the necessary methods for training, evaluating, and predicting.

### 2. NeuralNetwork.cs (Neural Network Implementation)
#### Overview
This component is the core of the system, implementing a feedforward neural network.

#### Main Functionality
- Network initialization with configurable layers and neurons.
- Forward propagation to calculate network outputs.
- Backpropagation to update weights using gradient descent.

### 3. Preprocessing.cs (Data Pre-processing)
#### Overview
This script deals with loading and pre-processing the data required for training the neural network.

#### Main Functionality
- Loading property data from CSV files.
- Handling missing values and data filtering.
- Standardization of features.
- Conversion of categorical variables into numerical representations.

### 4. Math.cs (Mathematical Functions)
#### Overview
Contains mathematical functions – in particular, the script contains implementation for activation functions and their derivatives.

#### Main Functionality
- Building the interface for activation functions.
- Implementation of such activation functions as ReLU (Rectified Linear Unit) and Identity Function.
- Implementation of activation functions derivatives.

### 5. Extensions.cs (Extension Methods)
#### Overview
This file contains extension methods to enhance existing classes.

#### Main Functionality
- A `Batch` method for partitioning enumerable collections into batches of a specified size.

### 6. UserOptions.cs (User Interaction)
#### Overview
Methods for processing user requests, interacting with users, and configuring, training, evaluating, and predicting with the help of the neural network.

#### Main Functionality
- Configuring neural network parameters (e.g., hidden layers, learning rate).
- Training the neural network with batch processing.
- Evaluating the model’s performance using various metrics (precision, recall, F1-score).
- Predicting property classes based on the user input.

## Conclusion

The Real Estate Prediction System is an experimental project aimed at predicting property classes using a neural network built from scratch in C#. The goal was to apply various skills including asynchronous programming, Object-Oriented Programming, data processing from CSV files, usage of LINQ (Language-Integrated Query), and the utilization of iterator and extension methods in C#.

In terms of performance, for a 5-class classification the system achieves an average precision, recall, and F1-score of around 50%. While this accuracy is not ideal, it represents a significant improvement over random guessing, which would achieve around 20% accuracy for a 5-class classification. Further adjustments and improvements to the network could lead to even more accurate predictions.

The system's design ensures that it can be maintained and extended easily, offering flexibility for future development.
