import numpy as np

from src.modules.read_data import read_data
from src.modules.save_data import save_points
from src.modules.dense_net import DenseNet, DenseLayer, ASigmoid, LQuadratic
from src.modules.black_box_adversary import BlackBoxAdversary

# Read in data once to reference for the following tests.
train_data, val_data, test_data = read_data()

def test_black_box_gen():
    # First create and train a model to work against
    layers = [DenseLayer(
        train_data[0][0].shape[0], 50), ASigmoid(), 
        DenseLayer(50, 10), ASigmoid(),
    ]
    loss = LQuadratic()
    net = DenseNet(layers=layers, loss=loss)

    n = 1000
    indices = np.random.permutation(train_data[0].shape[0])
    train_X, train_Y = train_data[0][indices[:n]], train_data[1][indices[:n]]
    val_X, val_Y = val_data[0][:4000], val_data[1][:4000]

    net.train(
        train_X=train_X, train_Y=train_Y,
        val_X=val_X, val_Y=val_Y,
        epochs=100, batch_size=200,
    )
    
    # Find a correctly classified image
    x, y = None, None
    for i in range(100):
        if np.argmax(net.predict(train_X[i])) == np.argmax(train_Y[i]):
            x = train_X[i]
            y = train_Y[i]
            break

    save_points([x], "black-box-target")

    adversary = BlackBoxAdversary(net=net, x=x, y=y)
    spook = adversary.generate()

    y_true = net.predict(x)
    y_spook = net.predict(spook)
    assert np.argmax(y_true) == np.argmax(y)
    assert np.argmax(y_spook) != np.argmax(y)
    save_points([spook], "generated-spook")
    print(f"Classifies target as {np.argmax(y_true)}")
    print(f"Classifies spook as {np.argmax(y_spook)}")
