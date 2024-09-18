"""Script containing nn layers and activations for the neural network classifier.

This script contains the implementation of the FullyConnected layer, which would be used
in all the hidden layers of the NeuralNetworkClassifier class. The script also contains
the implementation of the dropout regularization technique, the activation functions
together with their derivatives and the binary cross entropy loss function to all be
used in the NeuralNetworkClassifier class.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import Layer

logger = logging.getLogger(__name__)


class FullyConnected(Layer):
    def __init__(
        self, input_size: int, output_size: int, momentum: float = 0.9
    ) -> None:
        logger.debug("Initialising a Fully Connected Layer")
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.momentum: float = momentum
        super().__init__(input_size, output_size)

        # Randomise the weights and set the range to be between -0.5 and 0.5
        # np.rand is used as np.randn will give a normal distribution over a wider
        # range, potentially saturating weights early on into training.
        self.weights: np.ndarray = (
            np.random.rand(self.input_size, self.output_size) - 0.5
        )
        self.bias: np.ndarray = np.random.rand(1, self.output_size) - 0.5

        # Implement velocity for momentum. Velocity is used to store the running average
        # of the past gradients.
        self.velocity_weights: np.ndarray = np.zeros_like(self.weights)
        self.velocity_bias: np.ndarray = np.zeros_like(self.bias)

        logger.debug(
            f"Initialised Fully Connected Layer with input size {input_size} and output size {output_size}"
        )

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """Perform the forward pass through the layer.

        Parameters
        ----------
        input_data : np.ndarray
            Input data to the layer. Shape (batch_size, input_size)

        Returns:
        -------
        np.ndarray
            Output of the layer. Shape (batch_size, output_size)
        """
        logger.debug("Performing forward pass through the Fully Connected Layer")
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        logger.debug("Forward pass completed")
        return self.output

    def backward(self, output_error: np.ndarray, learning_rate: float) -> np.ndarray:
        """Perform the backward pass through the layer.

        Method first reshapes the output error if needed, then calculates the input
        error and weights error using the chain rule and the bias error by summing the
        output error. Velocity is then updated using the momentum and the accumulated
        gradients from the previous epochs. Weights and biases by adding the
        corresponding velocities.

        Parameters
        ----------
        output_error : np.ndarray
            Errors from the next layer in the network, used to calculate the input
            error. Shape (batch_size, output_size)
        learning_rate : float
            Learning rate of the neural network, used to update the weights and biases.

        Returns:
        -------
        np.ndarray
            Error to be passed to the previous layer in the network. Shape (batch_size,
            input_size)

        Notes:
        -----
        In vanilla gradient descent, the weights and biases are updated by simply
        subtracting the gradients multiplied by the learning rate. However, when using
        momentum, the weights and biases are updated by adding the velocity, because the
        velocity is an accumulation of the negative gradients over time. Adding the
        velocities is equivalent to subtracting the gradients.
        """
        logger.debug("Performing backward pass through the Fully Connected Layer")
        if output_error.ndim == 1:
            output_error = output_error.reshape(-1, self.output_size)

        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = output_error.sum()

        self.velocity_weights = (
            self.momentum * self.velocity_weights - learning_rate * weights_error
        )
        self.velocity_bias = (
            self.momentum * self.velocity_bias - learning_rate * bias_error
        )

        self.weights += self.velocity_weights
        self.bias += self.velocity_bias
        logger.debug("Backward pass completed")
        return input_error


def dropout(x: np.ndarray, dropout_rate: float) -> np.ndarray:
    """Implement dropout regularization.

    Dropout is a regularization technique that helps prevent overfitting by randomly
    setting a fraction of the neurons to 0 at each update during training. This would
    help to prevent the model from overfitting and improve generalization by allowing
    the model to learn more robust features in the training data.

    Parameters
    ----------
    x : np.ndarray
        Output from the previous layer.
    dropout_rate : float
        Probability of setting a neuron to 0.

    Returns:
    -------
    np.ndarray
        Output of the dropout layer.
    """
    logger.debug(f"Performing dropout with rate {dropout_rate}.")
    if dropout_rate == 0:
        logger.warning("No dropout applied as dropout rate is 0.")
        return x
    if dropout_rate < 0 or dropout_rate >= 1:
        raise ValueError("Dropout rate must be in the range [0, 1).")
    keep_prob = 1 - dropout_rate
    mask = np.random.rand(*x.shape) < keep_prob
    logger.debug("Dropout layer applied.")
    return x * mask


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function.

    ReLU is a non-linear activation function that is defined as the positive part of its
    argument. It is the most widely used activation function in deep learning models due
    to its simplicity, ability to learn complex patterns in the data and its speed in
    computation. Additionally, it's pretty robust to the vanishing gradient problem
    which gives it an edge over other activation functions like tanh.

    Parameters
    ----------
    x : np.ndarray
        Input to the activation function.

    Returns:
    -------
    np.ndarray
        Output of the activation function.
    """
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function.

    Sigmoid is a non-linear activation function that squashes the input to the range [0,
    1]. It is widely used in binary classification problems as it maps the input to a
    set of probabilities. Because of this, it is also used in the output layer of binary
    classification models.

    Parameters
    ----------
    x : np.ndarray
        Input to the activation function.

    Returns:
    -------
    np.ndarray
        Output of the activation function.
    """
    return 1 / (1 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation function.

    Tanh is a non-linear activation function that squashes the input to the range [-1,
    1]. It is similar to the sigmoid function but has a range of [-1, 1] which allows it
    to model negative values as well. This activation function is not used in the final
    implemented model, but it is included here for completeness and perhaps to be
    implemented if I have time to come back to it.

    Parameters
    ----------
    x : np.ndarray
        Input to the activation function.

    Returns:
    -------
    np.ndarray
        Output of the activation function.
    """
    return np.tanh(x)


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid function, for backpropagation."""
    sig = 1 / (1 + np.exp(-x))
    return sig * (1 - sig)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of the ReLU function, for backpropagation."""
    return np.where(x > 0, 1, 0)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of the tanh function, for backpropagation."""
    return 1 - np.tanh(x) ** 2


def binary_cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary Cross Entropy Loss.

    This loss function is used in binary classification problems to measure the
    difference between the true labels and the predicted probabilities. Before the loss
    is calculated, the predicted probabilities are clipped to avoid the log(0) case
    where the loss would be NaN.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred : np.ndarray
        Predicted probabilities.

    Returns:
    -------
    float
        Binary Cross Entropy Loss.
    """
    # Clip values to avoid the log(0) case which would result in NaN
    y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
