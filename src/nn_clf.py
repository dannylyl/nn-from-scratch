"""Neural Network Classifier Module Built with NumPy."""

import logging

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm
from typing import Tuple, List
from .base import BaseMLP
from .nn_utils import (
    FullyConnected,
    binary_cross_entropy_loss,
    dropout,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
)

logger = logging.getLogger(__name__)


class NeuralNetworkClassifier(BaseMLP):
    def __init__(self, params: DictConfig):
        """Neural Network Classifier Class Built with NumPy.

        The class inherits the ABC BaseMLP class and implements the abstract methods
        fit() and predict(). The class also implements the forward and backward methods,
        albeit a little hard coded for the specific architecture.

        Parameters
        ----------
        params : DictConfig
        A configuration object containing the following keys:
        - hidden_layer_sizes : int
            The number of neurons in each hidden layer. Defaults to 16.
        - batch_size : int
            The size of the mini-batches used during training. Defaults to 32.
        - learning_rate : float
            The learning rate used during gradient descent. Defaults to 0.01.
        - max_iter : int
            The maximum number of iterations (epochs) for training. Defaults to 50.
        - random_state : int
            The seed used for random number generation to ensure reproducibility.
            Defaults to 42.
        - momentum : float
            The momentum factor used to accelerate gradient descent. Defaults to 0.9.
        - input_size : int
            The number of input features.
        - output_size : int
            The number of output classes or units.
        - dropout : float
            The probability of dropping out neurons during training to prevent
            overfitting. Defaults to 0.2.

        Notes:
        -----
        This method also sets a random seed for reproducibility and calls the
        `build_model` method to construct the neural network layers.
        """
        logger.debug("Initialising Neural Network Classifier")
        self.hidden_layer_sizes: int = params.get("hidden_layer_sizes", 16)
        self.batch_size: int = params.get("batch_size", 32)
        self.learning_rate: float = params.get("learning_rate", 0.01)
        self.max_iter: int = params.get("max_iter", 50)
        self.random_state: int = params.get("random_state", 42)
        self.momentum: float = params.get("momentum", 0.9)
        super().__init__(
            self.hidden_layer_sizes,
            self.batch_size,
            self.learning_rate,
            self.max_iter,
            self.random_state,
            self.momentum,
        )
        np.random.seed(self.random_state)
        logger.debug(f"Set random seed to {self.random_state} for reproducibility.")
        # More parameters added on top of the base class
        self.input_size: int = params.get("input_size", 8)
        self.output_size: int = params.get("output_size", 1)
        self.dropout: float = params.get("dropout", 0.2)

        logger.debug("Successfully retrieved parameters for Neural Network Classifier.")

        # Build the neural net
        self.build_model()

    def build_model(self) -> None:
        """Method to build the neural network model with the specified architecture.

        The architecture is hard-coded to follow the specific structure in the
        assessment pdf file. The model consists of 4 hidden layers and 1 output layer.
        """
        logger.debug(
            "Building Neural Network Classifier model to specified architecture."
        )
        self.fc1 = FullyConnected(
            self.input_size, self.hidden_layer_sizes, self.momentum
        )
        self.fc2 = FullyConnected(
            self.hidden_layer_sizes, self.hidden_layer_sizes, self.momentum
        )
        self.fc3 = FullyConnected(
            self.hidden_layer_sizes, self.hidden_layer_sizes, self.momentum
        )
        self.fc4 = FullyConnected(
            self.hidden_layer_sizes, self.hidden_layer_sizes, self.momentum
        )
        self.output_layer = FullyConnected(
            self.hidden_layer_sizes, self.output_size, self.momentum
        )
        logger.debug("NeuralNetworkClassifier model built successfully.")
        pass

    def forward(self, x: np.ndarray, training: bool = True):
        """Perform the forward pass through the Neural Network Classifier.

        Computes the forward pass through the neural network layers and applies the
        dropout regularization technique between each hidden layer during training.

        After passing the data through each hidden layer, a ReLU activation function is
        applied to introduce non-linearity. The output layer uses a sigmoid activation
        to produce the final output.

        Additionally, a residual connection is implemented from the first hidden layer
        to the third hidden layer to improve the flow of gradients during training,
        according to the architecture specified in the assessment pdf file.

        Parameters
        ----------
        x : np.ndarray
            Input data to the neural network. Shape (batch_size, input_size)

        Returns:
        -------
        np.ndarray
            Output of the neural network. Shape (batch_size, output_size)
        """
        logger.debug("Forward method called for Neural Network Classifier.")
        if training:
            fc1_out = relu(self.fc1.forward(x))
            fc1_out = dropout(fc1_out, self.dropout)
            fc2_out = relu(self.fc2.forward(fc1_out))
            fc2_out = dropout(fc2_out, self.dropout)
            ## Implement residual connection from fc1_out to fc3_out ##
            fc3_out = relu(self.fc3.forward(fc2_out) + fc1_out)
            fc3_out = dropout(fc3_out, self.dropout)
            fc4_out = relu(self.fc4.forward(fc3_out))
            fc4_out = dropout(fc4_out, self.dropout)
            output = self.output_layer.forward(fc4_out)
        else:
            fc1_out = relu(self.fc1.forward(x))
            fc2_out = relu(self.fc2.forward(fc1_out))
            fc3_out = relu(self.fc3.forward(fc2_out) + fc1_out)
            fc4_out = relu(self.fc4.forward(fc3_out))
            output = self.output_layer.forward(fc4_out)
        logger.debug("Forward pass completed for Neural Network Classifier.")
        return sigmoid(output)

    def backward(self, out_err: np.ndarray) -> None:
        """Perform the backward pass through the Neural Network Classifier.

        Computes the backward pass, propagating the error gradients through the network
        and updating the weights and biases using gradient descent. The method applies
        the derivatives of the activation functions to calculate the gradients of the
        loss with respect to each layer's inputs and weights.

        Parameters
        ----------
        out_err : np.ndarray
            Error gradient from the output layer. Shape (batch_size, output_size)
        """
        logger.debug("Backward method called for Neural Network Classifier.")
        out_err = out_err * sigmoid_derivative(self.output_layer.output.squeeze())
        out_err = self.output_layer.backward(out_err, self.learning_rate)
        out_err = out_err * relu_derivative(self.fc4.output.squeeze())
        out_err = self.fc4.backward(out_err, self.learning_rate)
        out_err = out_err * relu_derivative(self.fc3.output.squeeze())
        out_err = self.fc3.backward(out_err, self.learning_rate)
        out_err = out_err * relu_derivative(self.fc2.output.squeeze())
        out_err = self.fc2.backward(out_err, self.learning_rate)
        out_err = out_err * relu_derivative(self.fc1.output.squeeze())
        out_err = self.fc1.backward(out_err, self.learning_rate)
        logger.debug("Backward pass completed for Neural Network Classifier.")
        pass

    def fit(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray] = None,
    ) -> None:
        """Fit the Neural Network Classifier to the training data.

        Main method to train the neural network model on the training data. The method
        calls the forward and backward methods for each epoch and mini-batch to update
        the weights and biases of the network. The training loss is calculated at the
        end of each epoch and stored in the `epoch_losses` list for monitoring after
        training.

        If validation data is provided, the method also evaluates the model on the
        validation data after each epoch and stores the validation loss in the
        `val_losses` list.

        Parameters
        ----------
        train_data : Tuple[np.ndarray, np.ndarray]
            A tuple containing the training features and labels. The features should
            have shape (num_samples, input_size) and the labels should have shape
            (num_samples, output_size).
        val_data : Tuple[np.ndarray, np.ndarray], optional
            A tuple containing the validation features and labels. The features should
            have shape (num_samples, input_size) and the labels should have shape
            (num_samples, output_size). Defaults to None.

        Notes:
        -----
        The method uses the binary cross-entropy loss function to calculate the loss
        within the training loop. I wrote the method to mirror a PyTorch-like training
        loop as closely as possible since that is what I am most familiar with.

        The method does not return anything, but the training and validation losses can
        be called as attributes of the class after training.
        """
        self.epoch_losses = []
        self.val_losses = []

        # Training loop
        for epoch in tqdm(range(self.max_iter)):
            batch_loss = []
            for i, batch in enumerate(range(0, len(train_data[0]), self.batch_size)):
                x, y = train_data
                x_batch = x[batch : batch + self.batch_size]
                y_batch = y[batch : batch + self.batch_size]
                y_batch = y_batch.squeeze()
                output = self.forward(x_batch)
                output = output.squeeze()
                loss = binary_cross_entropy_loss(y_batch, output)
                loss_grad = output - y_batch
                self.backward(loss_grad)
                batch_loss.append(loss)
            epoch_loss = np.mean(batch_loss)
            self.epoch_losses.append(epoch_loss)
            logger.info(f"Epoch {epoch+1} Training Loss:{epoch_loss}")

            # Validation loop if validation data is provided
            if val_data is not None:
                val_batch_loss = []
                for i, batch in enumerate(range(0, len(val_data[0]), self.batch_size)):
                    x_val, y_val = val_data
                    x_val_batch = x_val[batch : batch + self.batch_size]
                    y_val_batch = y_val[batch : batch + self.batch_size]
                    y_val_batch = y_val_batch.squeeze()
                    output_val = self.forward(x_val_batch, training=False)
                    output_val = output_val.squeeze()
                    loss_val = binary_cross_entropy_loss(y_val_batch, output_val)
                    val_batch_loss.append(loss_val)
                val_epoch_loss = np.mean(val_batch_loss)
                self.val_losses.append(val_epoch_loss)
                logger.info(f"Epoch {epoch+1} Validation Loss:{val_epoch_loss}")
        pass

    def predict(self, x: np.ndarray) -> List:
        """Perform predictions using the Neural Network Classifier.

        Perform predictions on the input data using the trained neural network model.
        The method calls the forward method to compute the output of the model and then
        returns the predictions after applying the sigmoid activation function.

        Parameters
        ----------
        x : np.ndarray
            Array of input data to make predictions on. Shape (num_samples, input_size)

        Returns:
        -------
        np.ndarray
            Prediction probabilities made by the model. Shape (num_samples, output_size)

        Notes:
        -----
        The method uses the sigmoid activation function to convert the output of the
        model to probabilities. For a binary classification task, the output will have
        to be subsequently thresholded to obtain the final class predictions.
        """
        all_predictions = []
        for i in range(0, len(x), self.batch_size):
            x_batch = x[i : i + self.batch_size]
            output = self.forward(x_batch)
            all_predictions.extend(output)
        return all_predictions

    def __str__(self) -> str:
        """A string representation of the Neural Network Classifier.

        Used to print the model summary when the object is printed.

        Returns:
        -------
        str
            A string representation of the summary of the model.
        """
        summary = "NeuralNetworkClassifier(\n"
        summary += f"  Input Layer: {self.input_size} -> {self.hidden_layer_sizes}\n"
        summary += f"  Hidden Layer 1: {self.hidden_layer_sizes} -> {self.hidden_layer_sizes}\n"
        summary += f"  Hidden Layer 2: {self.hidden_layer_sizes} -> {self.hidden_layer_sizes}\n"
        summary += f"  Hidden Layer 3: {self.hidden_layer_sizes} -> {self.hidden_layer_sizes} (with Residual Connection from Hidden Layer 1)\n"
        summary += f"  Hidden Layer 4: {self.hidden_layer_sizes} -> {self.hidden_layer_sizes}\n"
        summary += f"  Output Layer: {self.hidden_layer_sizes} -> {self.output_size}\n"
        summary += f"  Dropout Probability: {self.dropout}\n"
        summary += f"  Learning Rate: {self.learning_rate}\n"
        summary += f"  Momentum: {self.momentum}\n"
        summary += f"  Max Iterations: {self.max_iter}\n"
        summary += f"  Batch Size: {self.batch_size}\n"
        summary += f"  Random State: {self.random_state}\n"
        summary += ")"
        return summary

    def __repr__(self) -> str:
        return self.__str__()
