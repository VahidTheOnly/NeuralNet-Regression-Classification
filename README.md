## Neural Network Models with NumPy and TensorFlow

### Overview

This project comprises three Jupyter notebooks, each showcasing different neural network architectures for various tasks:

1. **Single Perceptron for Regression**
2. **Two-Layer Neural Network for Binary Classification**
3. **Multi-Layer Neural Network for Multi-Class Classification**

### `linear_regression_single_perceptron.ipynb`

This notebook illustrates a neural network model utilizing a single perceptron for linear regression tasks. Two variants are presented: one with a single input feature and another with two input features.

#### Contents
1. **Single Input Perceptron**
    - **Data Generation**: Synthetic data is created using `make_regression` from sklearn.
    - **Model Implementation**:
        - `initialize_parameters`: Initializes weights and biases.
        - `forward_propagation`: Computes the predicted output.
        - `compute_cost`: Calculates the mean squared error cost.
        - `gradient_descent`: Updates parameters using gradient descent.
        - `nn_model`: Trains the model using the above functions.
    - **Visualization**: Plots the regression line and data points.
    
2. **Two Input Perceptron**
    - **Data Preparation**: Reads and preprocesses the house prices dataset.
    - **Model Implementation**: Reuses functions from the single input model.
    - **Visualization and Evaluation**: Plots the regression results and calculates RMSE and R² score.
    - **Evaluation**: Calculates and displays the RMSE and R² score.

### `NeuralNet_with_Two_Layers.ipynb`

This notebook implements a neural network with one hidden layer for binary classification tasks. The hidden layer can have an arbitrary number of neurons.

#### Contents
1. **Data Generation**: Synthetic data is created using `make_blobs` from sklearn.
2. **Model Implementation**:
    - `initialize_parameters`: Initializes weights and biases for both layers.
    - `forward_propagation`: Computes the predicted output.
    - `compute_cost`: Calculates the binary cross-entropy loss.
    - `gradient_descent`: Updates parameters using gradient descent.
    - `nn_model`: Trains the model using the above functions.
    - `predict`: Makes predictions using the trained model.
    - `plot_decision_boundary`: Visualizes the decision boundary of the trained model.
3. **Visualization**: Plots decision boundaries for different datasets.

### `multi_layer_nn.ipynb`

This notebook implements a multi-layer neural network for multi-class classification tasks using the MNIST digits dataset.

#### Contents
1. **Data Preparation**:
    - Loads and preprocesses the digits dataset from scikit-learn.
    - Scales features using `MinMaxScaler`.
    - Splits the data into training and testing sets.
2. **Model Implementation**:
    - `initialize_parameters`: Initializes weights and biases for each layer.
    - `forward_propagation`: Computes the predicted output using softmax activation for the final layer.
    - `compute_cost`: Calculates the categorical cross-entropy loss.
    - `gradient_descent`: Updates parameters using gradient descent.
    - `learning_rate_decay`: Implements learning rate decay over epochs.
    - `mini_batch`: Creates mini-batches from data (X, y).
    - `nn_model`: Trains the model using the above functions.
    - `predict`: Makes predictions using the trained model.
3. **Evaluation**:
    - Evaluates the model using classification report and accuracy score first with learning rate decay and then with mini-batch.
    - Displays confusion matrix.
    - Visualizes misclassified examples.
