import numpy as np

from src.modules.read_data import read_data
from src.modules.save_data import save_points
from src.modules.dense_net import DenseNet, DenseLayer, ASigmoid, LQuadratic
from src.modules.black_box_adversary import BlackBoxAdversary

# Read in data once to reference for the following tests.
train_data, val_data, test_data = read_data()

def train_small_model():
    # Create and train a small neural net on a sample of training data. Returns the
    # network, and the training data it used.
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
    return net, train_X, train_Y

def test_black_box_gen():
    """
    Test that the black box adversarial generation can successfully generate a spook image
    which gets incorrectly classified, starting from a given correctly-classified image.
    """
    net, train_X, train_Y = train_small_model()
    
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

def test_black_box_gen():
    """
    Test that we can generate a new set of data all visually close to the correctly classified
    images. However, at the end we should have 100 new images all within a distance of at most 1
    from the original images, and the trained network should misclassify all of them (accuracy 0).
    """
    net, train_X, train_Y = train_small_model()

    new_set_size = 5
    spook_set_X = []
    spook_set_Y = []
    max_spook_dist = 1 # Don't accept spooks further than 1 in L-2 norm from a valid datapoint.
    
    i = 0
    while len(spook_set_X) < new_set_size:
        i += 1
        
        x, y = None, None
        if np.argmax(net.predict(train_X[i])) == np.argmax(train_Y[i]):
            x = train_X[i]
            y = train_Y[i]
        else:
            continue

        # x, y contains a "correctly classified" image point
        adversary = BlackBoxAdversary(net=net, x=x, y=y)

        # generate a new spook, which must be within a specific distance to x
        spook = None
        for _ in range(5):
            spook = adversary.generate()
            if np.linalg.norm(x - spook) > max_spook_dist:
                continue
            else:
                spook_set_X.append(spook)
                spook_set_Y.append(y)
                break
    
    spook_set_X = np.array(spook_set_X)
    spook_set_Y = np.array(spook_set_Y)

    print(net.get_accuracy(spook_set_X, spook_set_Y))
    # Uncomment below to save the spook data as images
    # save_points(spook_set_X, "spook")
    print(np.argmax(spook_set_Y, axis=1))
