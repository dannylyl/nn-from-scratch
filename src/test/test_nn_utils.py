"""Pytests for nn_utils.py"""

import pytest
import numpy as np
from omegaconf import DictConfig
from typing import Tuple
from .. import nn_utils


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


def test_fully_connected_initialisation(params):
    np.random.seed(params.random_state)
    fc = nn_utils.FullyConnected(
        input_size=params.input_size,
        output_size=params.output_size,
        momentum=params.momentum,
    )
    assert fc.input_size == params.input_size
    assert fc.output_size == params.output_size
    assert fc.momentum == params.momentum
    assert fc.weights.shape == (params.input_size, params.output_size)
    assert fc.bias.shape == (1, params.output_size)


def test_forward(params, input_data):
    np.random.seed(params.random_state)
    fc = nn_utils.FullyConnected(
        input_size=params.input_size,
        output_size=params.output_size,
        momentum=params.momentum,
    )
    x, _ = input_data
    fc.weights = np.array([[0.5], [0.5]])
    fc.bias = np.array([0.1])
    out = fc.forward(x)
    assert out.shape == (x.shape[0], params.output_size)
    assert np.allclose(out, np.array([[0.5], [-0.4], [0.65], [-0.8]]))


def test_dropout():
    x = np.array([[0.2, 0.6], [-0.3, -0.7], [0.3, 0.8], [-0.8, -1]])
    out = nn_utils.dropout(x, 0.5)
    assert out.shape == x.shape
    assert np.sum(out) != 0


def test_relu():
    x = np.array([[0.2, 0.6], [-0.3, -0.7], [0.3, 0.8], [-0.8, -1]])
    out = nn_utils.relu(x)
    assert out.shape == x.shape
    assert np.allclose(out, np.array([[0.2, 0.6], [0, 0], [0.3, 0.8], [0, 0]]))


def test_sigmoid():
    x = np.array([[0.2, 0.6], [-0.3, -0.7], [0.3, 0.8], [-0.8, -1]])
    out = nn_utils.sigmoid(x)
    assert out.shape == x.shape
    assert np.all(out >= 0) and np.all(out <= 1)


def test_bce_loss():
    y_true = np.array([[0], [1], [1], [0]])
    y_pred = np.array([[0.2], [0.8], [0.9], [0.1]])
    loss = nn_utils.binary_cross_entropy_loss(y_true, y_pred)
    print(loss)
    assert loss > 0
