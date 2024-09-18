"""Pytests for nn_clf module"""

import pytest
import numpy as np
from omegaconf import DictConfig
from typing import Tuple, List
from ..nn_clf import NeuralNetworkClassifier


@pytest.fixture
def params() -> DictConfig:
    params = {
        "hidden_layer_sizes": 2,
        "batch_size": 2,
        "learning_rate": 0.01,
        "max_iter": 10,
        "random_state": 42,
        "momentum": 0.9,
        "input_size": 2,
        "output_size": 1,
        "dropout": 0.2,
    }
    return DictConfig(params)


@pytest.fixture
def input_data() -> Tuple[np.ndarray, np.ndarray]:
    return np.array([[0.2, 0.6], [-0.3, -0.7], [0.3, 0.8], [-0.8, -1]]), np.array(
        [[0], [1], [1], [0]]
    )


def test_initialisation(params):
    model = NeuralNetworkClassifier(params)
    assert model.hidden_layer_sizes == params.hidden_layer_sizes
    assert model.batch_size == params.batch_size
    assert model.learning_rate == params.learning_rate
    assert model.max_iter == params.max_iter
    assert model.random_state == params.random_state
    assert model.momentum == params.momentum
    assert model.input_size == params.input_size
    assert model.output_size == params.output_size
    assert model.dropout == params.dropout


def test_forward(params, input_data):
    model = NeuralNetworkClassifier(params)
    x, _ = input_data
    out = model.forward(x)
    assert out.shape == (x.shape[0], params.output_size)


def fit(params, input_data):
    model = NeuralNetworkClassifier(params)
    initial_weights = model.fc1.weights.copy()
    model.fit(input_data)
    end_weights = model.fc1.weights.copy()

    assert not np.allclose(initial_weights, end_weights)
    assert len(model.epoch_losses) == params.max_iter


def test_predict(params, input_data):
    model = NeuralNetworkClassifier(params)
    x, _ = input_data
    model.fit(input_data)
    y_pred = model.predict(x)
    assert len(y_pred) == x.shape[0]
    assert isinstance(y_pred, List)
