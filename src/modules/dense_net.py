# Defines classes needed to make a simple, dense neural network.
# 
# Terms:
#  - W (weights) [input-layer, output-layer]
#  - b (biases) [output-layer, 1]
#  - X (Batch of inputs - shape[batch-size, input-layer])
#  - Z (Result of { W @ X + b })
#  - A (Result of applying Activation function to Z)

import numpy as np

class ALinear():
    """
    Simple linear activation function. Simple passes the data through.
    """

    def __init__(self):
        pass

    def forward(self, Z):
        return Z

class LQuadratic():
    """
    Loss function through the quadratic (square loss) function.
    """

    def __init__(self):
        pass

    def calc(y_pred, y_true):
        """Returns the loss on a given batch."""

        n = y_pred.shape[0]
        return (1 / 2*n) * (np.linalg.norm(y_pred - y_true, ord=2) ** 2)

    def calc_gradient():
        """Returns the initial gradient to kick off gradient descent."""
        pass

class DenseLayer():
    """
    Simple dense layer with "n" neurons. It needs to know how many connections are coming in
    from the prior layer.
    """
    
    def __init__(self, in_size, out_size):
        """
        Initialize the layer with random values for the initial weights and params.

        :param in_size: The amount of nodes in the prior 
        """
        self.W = np.random.random((in_size, out_size))
        self.b = np.random.random(1, out_size)

    def forward(self, X):
        """Performs the forward pass."""
        return X @ self.W + self.b


class DenseNet():
    def __init__(self, layers):
        self.layers = layers

    def train(train_X, train_Y, epochs=10, batch_size=1):
        """Takes in X, Y, epochs, and batch size."""

        # Create mini-batches

        # Forward pass through self.layers

        # Calculate loss

        # Calculate initial gradient

        # Backwards pass through reversed(self.layers)

        # Update weights
        
        pass
