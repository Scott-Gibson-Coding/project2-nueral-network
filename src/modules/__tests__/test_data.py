import numpy as np

from src.modules.read_data import read_data, one_hot_encode

# Read in data once to reference for the following tests.
train_data, val_data, test_data = read_data()

def test_input_data_groups():
    """Expect input data to be split into three groups, training, val, testing."""
    assert len(train_data) == 2
    assert len(val_data) == 2
    assert len(test_data) == 2

def test_input_data_shape():
    """Expect each input dataset in/out to have a consistent shape."""
    assert train_data[0].shape[0] == train_data[1].shape[0]
    assert val_data[0].shape[0] == val_data[1].shape[0]
    assert test_data[0].shape[0] == test_data[1].shape[0]

def test_input_one_hot_encoded():
    """Expect output vals to be one hot encoded, so 2 should be [0, 0, 1, 0, ...]"""
    assert train_data[1].shape[1] == 10
    assert val_data[1].shape[1] == 10
    assert test_data[1].shape[1] == 10

def test_can_decode_output():
    """Should be able to decode some output data into the guessed values."""
    Y = np.zeros((5, 10))
    Y[0][0] = 1
    Y[1][3] = 1
    Y[2][7] = 1
    Y[3][2] = 1
    Y[4][9] = 1

    Y_decoded = one_hot_encode(Y, decode=True)
    assert np.array_equal(Y_decoded, [0, 3, 7, 2, 9])

def test_input_img():
    """Expect each data point to be able to contain a 28x28 img."""
    img_size = 28 * 28
    assert train_data[0].shape[1] == img_size
