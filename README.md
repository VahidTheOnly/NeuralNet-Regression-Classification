# Neural Network Models for Regression and Classification

This repository contains Python code implementing neural network models for regression and binary classification and also multiclass classification tasks. The neural networks are built from scratch using Python and TensorFlow for computing derivatives.

## Files Included

1. `linear_regression_single_perceptron.ipynb`: This Jupyter notebook contains the implementation of a single perceptron neural network model for linear regression. It demonstrates two variations of the model: one with a single input node and the other with two input nodes.

2. `NeuralNet_with_Two_Layers.ipynb`: This Jupyter notebook implements a neural network model with two layers for binary classification. The network architecture consists of an input layer with two nodes, a hidden layer with two or more perceptrons, and an output layer with one perceptron.

3. `multi_layer_nn.ipynb`: This Jupyter notebook implements a neural network model with an arbitrary number of layers multiclass classification.

## Dataset

### Regression Task:
For the regression task, synthetic data is generated using `make_regression` from sklearn. Additionally, real estate data from a CSV file named 'house_prices_train.csv' is utilized.

### Binary Classification Task:
For the binary classification task, synthetic data is created using `make_blobs` from sklearn. and also two simple dataset 'Arcs.csv' and 'flower.csv' are utilized.

### MultiClass Classification Task:
For the multiclass classification task, `Digits dataset` loaded from sklearn. similar to mnist dataset it contains images of numbers between Zero to Ten.

## Functionality

- `initialize_parameters`: Initializes the weights and biases of the neural network layers.
- `forward_propagation`: Performs forward propagation through the network layers.
- `compute_cost`: Calculates the cost (loss) function.
- `gradiant_descent`: Updates the parameters using gradient descent optimization.
- `nn_model`: Implements the training process of the neural network models.
- `predict`: Predicts output labels for input data.
- `plot_decision_boundary`: Visualizes the decision boundary for the classification model.


