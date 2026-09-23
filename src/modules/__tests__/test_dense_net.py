from pytest import approx
import numpy as np

from src.modules.read_data import read_data, one_hot_encode
from src.modules.dense_net import DenseNet, DenseLayer, ALinear, LQuadratic

# Read in data once to reference for the following tests.
train_data, val_data, test_data = read_data()

### TEST ACTIVATION CLASSES ###

### TEST LOSS CLASSES ###

def test_lquadratic():
    """Expect the square-loss to be 0 when outputs are equal, and large when messed up."""
    L = LQuadratic()

    pred = np.array([[0, 0, 0, 0, 1]])
    actual = np.array([[0, 0, 0, 0, 1]])

    assert L.calc(pred, actual) == 0

    pred = np.array([[1, 0, 0, 0, 0]])
    actual = np.array([[0, 0, 0, 0, 1]])

    assert L.calc(pred, actual) == approx(1)

    pred = np.array([
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
    ])
    actual = np.array([
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
    ])

    assert L.calc(pred, actual) == approx(1)

### TEST TRAINING ###
def get_training_net(hidden_layer_neurons=10):
    input_size = train_data[0][0].shape[0]
    n = hidden_layer_neurons

    layers = [
        DenseLayer(input_size, n), # Dense layer with 30 neurons
        ALinear(), # Linear activation function, temporary
        DenseLayer(n, 10), # Dense output layer, 10 neurons
    ]
    loss = LQuadratic()

    return DenseNet(layers=layers, loss=loss)

def test_predict_single_value():
    """Should be able to get predictions for a single value"""

    net = get_training_net(30)
    test_X = test_data[0]
    test_Y = test_data[1]
    np.argmax()
