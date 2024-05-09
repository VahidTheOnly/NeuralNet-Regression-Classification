# Neural Network Models for Regression and Classification

This repository contains Python code implementing neural network models for both regression and binary classification tasks. The neural networks are built from scratch using Python and TensorFlow for computing derivatives.

## Files Included

1. `linear_regression_single_perceptron.ipynb`: This Jupyter notebook contains the implementation of a single perceptron neural network model for linear regression. It demonstrates two variations of the model: one with a single input node and the other with two input nodes.

2. `NeuralNet_with_Two_Layers.ipynb`: This Jupyter notebook implements a neural network model with two layers for binary classification. The network architecture consists of an input layer with two nodes, a hidden layer with two perceptrons, and an output layer with one perceptron.

## Dataset

### Regression Task:
For the regression task, synthetic data is generated using `make_regression` from sklearn. Additionally, real estate data from a CSV file named 'house_prices_train.csv' is utilized.

### Classification Task:
For the binary classification task, synthetic data is created using `make_blobs` from sklearn.

## Network Architecture

### Regression Model:
- **Input Layer:** One or two input nodes, depending on the variant.
- **Output Layer:** One output node.
- **Activation Function:** Linear activation function.

### Classification Model:
- **Input Layer:** Two input nodes.
- **Hidden Layer:** One hidden layer with two perceptrons.
- **Output Layer:** One output node.
- **Activation Function:** Sigmoid activation function for both the hidden and output layers.

## Training Process

The models are trained using gradient descent optimization. For each epoch, the cost (loss) is calculated and printed to monitor the training progress.

## Functionality

- `initialize_parameters`: Initializes the weights and biases of the neural network layers.
- `forward_propagation`: Performs forward propagation through the network layers.
- `compute_cost`: Calculates the cost (loss) function.
- `gradiant_descent`: Updates the parameters using gradient descent optimization.
- `nn_model`: Implements the training process of the neural network models.
- `predict`: Predicts output labels for input data.
- `plot_decision_boundary`: Visualizes the decision boundary for the classification model.


