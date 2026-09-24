from pytest import approx
import numpy as np

from src.modules.read_data import read_data, one_hot_encode
from src.modules.dense_net import DenseNet, DenseLayer, ASigmoid, LQuadratic

# Read in data once to reference for the following tests.
train_data, val_data, test_data = read_data()

### TEST ACTIVATION CLASSES ###

def test_sigmoid():
    """Test that the sigmoid activation function is behaving (roughly) as expected."""
    act = ASigmoid()

    a_out = act.forward(np.array([-400, -5, 0, 5, 400]))

    assert a_out[0] == approx(0)
    assert a_out[1] > 0.001 and a_out[1] < 0.01
    assert a_out[2] == approx(0.5)
    assert a_out[3] > 0.990 and a_out[3] < 0.999
    assert a_out[4] == approx(1)

def test_sigmoid_matrix():
    """Test that the sigmoid activation function is behaving (roughly) as expected on a matrix."""
    act = ASigmoid()

    a_in = np.array([[-2, 2, 3.5, -4, 0], [1, 1.1, 2.3, 4.7, -5]])
    a_out = act.forward(a_in)

    assert a_out.shape == a_in.shape
    assert np.all((a_out >= 0) & (a_out <= 1))

### TEST LOSS CLASSES ###

def test_lquadratic():
    """Expect the square-loss to be 0 when outputs are equal, and large when messed up."""
    L = LQuadratic()

    pred = np.array([[0, 0, 0, 0, 1]])
    actual = np.array([[0, 0, 0, 0, 1]])

    assert L.forward(pred, actual) == 0

    pred = np.array([[1, 0, 0, 0, 0]])
    actual = np.array([[0, 0, 0, 0, 1]])

    assert L.forward(pred, actual) == approx(1)

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

    assert L.forward(pred, actual) == approx(1)

### TEST PREDICTION/VALIDATION ###

def get_training_net(hidden_layer_neurons=10):
    input_size = train_data[0][0].shape[0]
    n = hidden_layer_neurons

    layers = [
        DenseLayer(input_size, n), # Dense layer with 30 neurons
        ASigmoid(), # Sigmoid activation function
        DenseLayer(n, 10), # Dense output layer, 10 neurons
        ASigmoid(), # Sigmoid activation function
    ]
    loss = LQuadratic()

    return DenseNet(layers=layers, loss=loss)

def test_predict_single_value():
    """Should be able to get predictions for a single value"""

    net = get_training_net(30)
    test_X = test_data[0]

    pred_Y = one_hot_encode(net.predict(test_X[5,:]), decode=True)
    assert pred_Y.shape[0] == 1
    assert np.all((pred_Y >= 0) & (pred_Y <= 9))

def test_predict_batch():
    """Should be able to get predictions for a batch of data points."""

    net = get_training_net(30)
    test_X = test_data[0]

    pred_Y = one_hot_encode(net.predict(test_X[:99,:]), decode=True)
    assert pred_Y.shape[0] == 99
    assert np.all((pred_Y >= 0) & (pred_Y <= 9))

def test_validation():
    """Should be able to get a risk score for a validation set."""

    net = get_training_net(30)
    val_X = val_data[0]
    val_Y = val_data[1]

    risk = net.validate(val_X, val_Y)
    assert risk > 0

def test_training_small_1_batch():
    """
    Should be able to train with the following:
      - Batch Size: 15
      - Epochs: 10
      - Training Data: 50 elements
      - Validation Data: Training Data
    """

    net = get_training_net(30)
    epochs = 10
    batch_size = 15
    train_X = train_data[0][:50,:]
    train_Y = train_data[1][:50,:]

    starting_risk = net.validate(train_X, train_Y)
    net.train(
        train_X=train_X, train_Y=train_Y,
        val_X=train_X, val_Y=train_Y,
        epochs=epochs,
        batch_size=batch_size
    )
    ending_risk = net.validate(train_X, train_Y)
    assert starting_risk > ending_risk

# Got %94.43
def test_full_training_set():
    """Expect reasonable accuracy testing on full dataset."""
    net = get_training_net(30)
    epochs = 20
    batch_size = 50

    starting_risk = net.validate(test_data[0], test_data[1])
    net.train(
        train_X=train_data[0], train_Y=train_data[1],
        val_X=val_data[0], val_Y=val_data[1],
        epochs=epochs,
        batch_size=batch_size
    )
    ending_risk = net.validate(test_data[0], test_data[1])
    assert starting_risk > ending_risk
    net.get_accuracy(test_data[0], test_data[1])

# Got %95.08 accuracy on the test data
def test_full_training_set_expensive():
    """Expect reasonable accuracy testing on full dataset."""
    net = get_training_net(100)
    epochs = 30
    batch_size = 100

    starting_risk = net.validate(test_data[0], test_data[1])
    net.train(
        train_X=train_data[0], train_Y=train_data[1],
        val_X=val_data[0], val_Y=val_data[1],
        epochs=epochs,
        batch_size=batch_size
    )
    ending_risk = net.validate(test_data[0], test_data[1])
    assert starting_risk > ending_risk

    net.get_accuracy(test_data[0], test_data[1])

# Managed ###.## risk
def test_overfit():
    """Expect reasonable accuracy testing on full dataset."""
    net = get_training_net(40)
    epochs = 50000
    batch_size = 1

    data_X = test_data[0][25:75]
    data_Y = test_data[1][25:75]

    starting_risk = net.validate(data_X, data_Y)
    net.train(
        train_X=data_X, train_Y=data_Y,
        val_X=data_X, val_Y=data_Y,
        epochs=epochs,
        batch_size=batch_size
    )
    ending_risk = net.validate(data_X, data_Y)
    assert starting_risk > ending_risk

    net.get_accuracy(data_X, data_Y)
