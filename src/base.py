from abc import ABC, abstractmethod


class BaseMLP(ABC):
    """Base class for MLP classification."""

    def __init__(
        self,
        hidden_layer_sizes,
        batch_size,
        learning_rate,
        max_iter,
        random_state,
        momentum,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.momentum = momentum

    @abstractmethod
    def fit(self, x, y) -> None:
        """Fit the model to data matrix X and target(s) Y.

        Parameters
        ----------
        x : list or sparse matrix of shape (n_samples, n_features)
            The input data
        y : list of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels for classification).
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def predict(self, x):
        """Predict using the multi-layer perceptron classifier.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data

        Returns:
        -------
        y : list, shape (n_samples) or  (n_samples, n_classes)
            The predicted classes.
        """
        raise NotImplementedError("Method not implemented")


class Layer(ABC):
    """Base class for Neural Network Layers. Added to this script."""

    def __init__(self, input, output):
        self.input = input
        self.output = output

    @abstractmethod
    def forward(self, input):
        """Forward pass through the layer.

        Parameters
        ----------
        input : np.ndarray
            Input data to the layer.

        Returns:
        -------
        np.ndarray
            Output data from the layer.
        """
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def backward(self, output_error, learning_rate):
        """Backward pass through the layer.

        Parameters
        ----------
        output_error : np.ndarray
            Error from the next layer in the network.
        learning_rate : float
            Learning rate of the neural network.

        Returns:
        -------
        np.ndarray
            Error to be passed to the previous layer in the network.
        """
        raise NotImplementedError("Method not implemented")
