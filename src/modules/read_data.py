import pickle
import numpy as np

from .constants import DATA_FILE_PATH

def load_data(file_path):
    """
    Depickles the requested file path using the python "pickle" library. Returns three
    sets of [input data points, output values].
    """
    with open(file_path, "rb") as file:
        in_data = pickle.load(file, encoding="latin1")
        train_data = [np.array(in_data[0][0]), np.array(in_data[0][1])]
        val_data = [np.array(in_data[1][0]), np.array(in_data[1][1])]
        test_data = [np.array(in_data[2][0]), np.array(in_data[2][1])]
    return train_data, val_data, test_data

def one_hot_encode(Y, decode=False):
    """
    Either encodes Y into one-hot encoded arrays of 1s and 0s, or decodes into the original
    integer values.
    """
    if decode == False:
        # One-hot encode
        Y_out = np.zeros((Y.shape[0], 10), dtype=np.int64)
        for i in range(Y.shape[0]):
            Y_out[i, Y[i]] = 1
        return Y_out
    else:
        return np.argmax(Y, axis=1)

def read_data():
    """
    Reads in the dataset stored in "digits.pkl". 
    
    :returns tuple: Three lists of grouped datasets. Each dataset has one nparray of inputs,
    and one nparray of output digits. Order is "training", "validation", "testing".
    """

    train_data, val_data, test_data = load_data(DATA_FILE_PATH)
    train_data[1] = one_hot_encode(train_data[1])
    val_data[1] = one_hot_encode(val_data[1])
    test_data[1] = one_hot_encode(test_data[1])

    return train_data, val_data, test_data
    