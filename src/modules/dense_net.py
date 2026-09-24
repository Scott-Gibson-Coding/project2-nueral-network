# Defines classes needed to make a simple, dense neural network.
# 
# Terms:
#  - W (weights) [input-layer, output-layer]
#  - b (biases) [output-layer, 1]
#  - X (Batch of inputs - shape[batch-size, input-layer])
#  - Z (Result of { W @ X + b })
#  - A (Result of applying Activation function to Z)
#
# Exposed functions:
#  - For consistency, layers and the cost/risk calculation function should expose a
#    "forward" function, and a "backwards" function.
#
# TODO Room for Improvement
# - Experiment with different hyperparameters
# - Look into adjusting step-size with "momentum", and early stop if the
#   cost diff between epochs slows down or fully stops
# - Implement some additional activation functions, like ReLU

import numpy as np

### LOSS/COST CLASSES

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

### ACTIVATION CLASSES

class ASigmoid():
    """
    Simple linear activation function. Simple passes the data through.
    """

    def __init__(self):
        self.Z = None # Cache for the input of Z passed into the layer.

    def _sigmoid(self, Z):
        """Performs sigmoid element-wise over the matrix Z."""

        # Clamp Z between -50 and 50 to avoid overflow
        
        return 1 / (1 + np.exp(-np.clip(Z, -50, 50)))
    
    def forward(self, Z):
        # Cache values for backwards pass
        self.Z = Z
        
        return self._sigmoid(Z)

    def backwards(self, chain_grad, step_size=None):
        # Multiplies the gradient passed in by the derivative of the activation.

        s = self._sigmoid(self.Z)
        d_sigmoid = s * (1 - s)
        return np.multiply(d_sigmoid, chain_grad)

### LAYER CLASSES

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
        self.W = np.random.rand(in_size, out_size) * 2 - 1
        self.b = np.random.rand(1, out_size) * 2 - 1
        self.X = None

    def forward(self, X):
        """Performs the forward pass, sends output to activation layer."""

        # Cache values for backwards pass
        self.X = X

        # Return linear combination of activations, weights, and biases
        return X @ self.W + self.b

    def backwards(self, chain_grad, step_size=1):
        """Performs a backwards pass, updating weights and biases before passing on next gradient."""

        # Forward gradient with respect to weights to prior activation
        next_layer_grad = chain_grad @ self.W.transpose()

        # Update weights, and biases
        self.W -= self.X.transpose() @ chain_grad * step_size
        self.b -= np.sum(chain_grad, axis=0) * step_size

        return next_layer_grad


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

    def get_accuracy(self, test_X, test_Y):
        """Returns the accuracy of the model on some test data."""
        X = np.copy(test_X)
        for layer in self.layers:
            X = layer.forward(X)

        for row in X:
            idx = np.argmax(row)
            row *= 0
            row[idx] = 1

        wrong_count = np.count_nonzero(test_Y - X) / 2
        accuracy = (test_X.shape[0] - wrong_count) / test_X.shape[0] * 100
        print(f"Accuracy of model on test dataset: % {round(accuracy, 3)}")
        return accuracy

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

    def _get_random_batches(self, X, Y, batch_size):
        """Generates a shuffled array of indices."""
        num_batches = X.shape[0]
        indices = np.random.permutation(num_batches)

        for i in range(0, num_batches, batch_size):
            batch_indices = indices[i:i+batch_size]
            yield X[batch_indices], Y[batch_indices]

        # In the case the batch_size doesn't divide evenly, use the last "batch_size" elements as the final batch.
        batch_indices = indices[-batch_size:]
        yield X[batch_indices], Y[batch_indices]

    def train(self, train_X, train_Y, val_X, val_Y, epochs=5, batch_size=1):
        """Takes in X, Y, epochs, and batch size."""

        for epoch in range(1, epochs+1):
            # Create mini-batches
            batch_gen = self._get_random_batches(train_X, train_Y, batch_size=batch_size)
            for batch_X, batch_Y in batch_gen:
                # Forward pass through self.layers
                X = batch_X
                for layer in self.layers:
                    X = layer.forward(X)

                # Calculate loss
                self.loss.forward(X, batch_Y)

                # Calculate initial gradient
                grad = self.loss.backwards()

                # Backwards pass through reversed(self.layers)
                for layer in reversed(self.layers):
                    grad = layer.backwards(grad, step_size=1)

            # Validate results and continue to next epoch
            self.validate(val_X, val_Y, epoch)
