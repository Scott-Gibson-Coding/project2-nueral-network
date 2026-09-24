# Defines classes needed to make a simple, dense neural network.
# 
# Terms:
#  - W (weights) [input-layer, output-layer]
#  - b (biases) [output-layer, 1]
#  - X (Batch of inputs - shape[batch-size, input-layer])
#  - Z (Result of { W @ X + b })
#  - A (Result of applying Activation function to Z)

import numpy as np

class ASigmoid():
    """
    Simple linear activation function. Simple passes the data through.
    """

    def __init__(self):
        self.Z = None # Cache for the input of Z passed into the layer.

    def _sigmoid(self, Z):
        """Performs sigmoid element-wise over the matrix Z."""
        return 1 / (1 + np.exp(-Z))
    
    def forward(self, Z):
        # Cache values for backwards pass
        self.Z = Z
        
        return self._sigmoid(Z)

    def backwards(self, chain_grad):
        # Multiplies the gradient passed in by the derivative of the activation.

        d_sigmoid = self._sigmoid(self.Z) * (1 - self._sigmoid(self.Z))
        return np.multiply(d_sigmoid, chain_grad)

class LQuadratic():
    """
    Loss function through the quadratic (square loss) function.
    """

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        """Returns the loss on a given batch."""

        # Cache values for backwards pass
        self.y_pred = y_pred
        self.y_true = y_true

        n = y_pred.shape[0]
        # Use Frobenius norm to get sum of squared differences
        return (np.linalg.norm(y_pred - y_true) ** 2) / (2 * n)

    def backwards(self):
        """Returns the initial gradient to kick off gradient descent."""

        n = self.y_pred.shape[0]

        # Avg gradient w.r.t. activated predictions
        return (self.y_pred - self.y_true) / n

class DenseLayer():
    """
    Simple dense layer with "n" neurons. It needs to know how many connections are coming in
    from the prior layer.
    """
    
    def __init__(self, in_size, out_size):
        """
        Initialize the layer with random values for the initial weights and params.

        :param in_size: The nodes sending information into this layer.
        :param out_size: The nodes in this layer which will send data to the following layer.
        """
        self.W = np.random.random((in_size, out_size))
        self.b = np.random.random((1, out_size))

    def forward(self, X):
        """Performs the forward pass, sends output to activation layer."""
        return X @ self.W + self.b

    def backwards(self):
        """Performs a backwards pass, updating weights and biases before passing on next gradient."""
        pass


class DenseNet():
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def predict(self, data):
        """Returns a matrix of predicted outputs."""
        X = np.copy(data)
        for layer in self.layers:
            X = layer.forward(X)
        
        return X

    def validate(self, val_X, val_Y, epoch=None):
        """Performs only a forwards pass and determines the loss on a validation set."""
        X = np.copy(val_X)
        for layer in self.layers:
            X = layer.forward(X)

        pred_Y = X
        risk = self.loss.forward(pred_Y, val_Y)

        if epoch == None:
            print(f"Loss on input data: {risk}")
        else:
            print(f"Epoch {epoch} --- {risk}")

        return risk

    def train(self, train_X, train_Y, val_X, val_Y, epochs=5, batch_size=1):
        """Takes in X, Y, epochs, and batch size."""

        for epoch in range(1, epochs+1):
            # Create mini-batches
            # TODO Temp mini batch is just the first 50 elements
            batch_X = train_X[:50,:]
            batch_Y = train_Y[:50,:]

            # Forward pass through self.layers
            X = batch_X
            for layer in self.layers:
                X = layer.forward(X)

            # Calculate loss
            risk = self.loss.forward(X, batch_Y)

            # Calculate initial gradient
            grad = self.loss.backwards()

            # Backwards pass through reversed(self.layers)
            for layer in reversed(self.layers):
                grad = layer.backwards(grad)

            # Update weights

            # Validate results and continue to next epoch
            # TODO Validating on the same training batch to see if its actually learning it
            self.validate(batch_X, batch_Y, epoch)
            # self.validate(val_X, val_Y, epoch)
