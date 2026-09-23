from pytest import approx
import numpy as np

from src.modules.read_data import read_data
from src.modules.dense_net import LQuadratic

# Read in data once to reference for the following tests.
# train_data, val_data, test_data = read_data()

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
