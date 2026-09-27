import numpy as np

from src.modules.read_data import read_data
from src.modules.dense_net import DenseNet, DenseLayer, ASigmoid, LQuadratic

# Read in data once to reference for the following tests.
train_data, val_data, test_data = read_data()

def convert_index_to_binary(Y):
    """
    Converts data encoded as a 10-vector with 1 for the index that represents the number,
    to a 4-vector with a binary encoding for the number. So 3 --> 0 0 1 1, 9 --> 1 0 0 1.
    """

    indices = np.argmax(Y, axis=1)
    shifts = np.arange(3, -1, -1)
    return (indices[..., None] >> shifts) & 1

# Test modifying the output layer to support 4 neurons
def test_bit_shifts():
    train_X = train_data[0][:50000]
    train_Y = convert_index_to_binary(train_data[1][:50000])
    val_X = val_data[0]
    val_Y = convert_index_to_binary(val_data[1])
    test_X = test_data[0]
    test_Y = convert_index_to_binary(test_data[1])

    print("Example of encoded training labels:")
    print(train_data[1][:10])
    print(train_Y[:10])

    # Create neural net
    layers = [DenseLayer(
        train_data[0][0].shape[0], 60), ASigmoid(), 
        DenseLayer(60, 4), ASigmoid(), # Note, four neurons in output layer
    ]
    loss = LQuadratic()
    net = DenseNet(layers=layers, loss=loss)

    net.train(
        train_X=train_X, train_Y=train_Y,
        val_X=val_X, val_Y=val_Y,
        epochs=100, batch_size=250, verbose=True,
    )

    # TODO My get_accuracy() function makes the assumption that the output layer will
    # only try to maximize one value. So if you call get_accuracy() with this binary
    # encoded set, you'll get very low accuracy percentage values which misrepresent
    # the ability of the trained net.

    # Manually determining accuracy, expect it to exceed at least 90%.
    # Need a heuristic to tell between a 0 and a 1, setting it to 0.8 for now.
    threshold = 0.8
    pred_Y = net.predict(test_X)
    pred_Y[pred_Y > threshold] = 1
    pred_Y[pred_Y <= threshold] = 0
    correct = 0
    total = pred_Y.shape[0]
    for i in range(total):
        if np.allclose(pred_Y[i], test_Y[i]):
            correct += 1
    accuracy = correct / total * 100
    print(f"Test accuracy: {accuracy}%")
