"""
A simple, fully-connected policy network.

Architecture (fixed by the assignment):
    input (k units)  ->  hidden1 (ReLU)  ->  hidden2 (ReLU)  ->  output (4 units, softmax)

k = number of sensors, hidden sizes <= 100 units each.
The 4 outputs are probabilities over the actions:
    0 = turn left, 1 = turn right, 2 = speed up, 3 = no action

This is written with plain NumPy and basic gradient descent, no extra
tricks (no Adam, no early stopping) to keep things easy to follow.
"""
import numpy as np

SEED = 42


def relu(z):
    return np.maximum(0.0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def softmax(z):
    # subtract the row max first, just for numerical stability
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


class PolicyNetwork:
    def __init__(self, k, hidden1=32, hidden2=32, seed=SEED):
        """k: number of sensor inputs. hidden1/hidden2: hidden layer sizes (max 100)."""
        assert 1 <= hidden1 <= 100 and 1 <= hidden2 <= 100, "hidden layers must be 1-100 units"

        rng = np.random.default_rng(seed)
        # the network's only input is the k sensor readings
        sizes = [k, hidden1, hidden2, 4]

        self.W = []
        self.b = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # small random weights, scaled down so training starts stable
            self.W.append(rng.normal(0.0, 0.1, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))

    def forward(self, X):
        """Runs X through the network. Returns the pre-activations and
        activations of every layer, so backward() can reuse them."""
        pre_activations = []
        activations = [X]
        a = X
        last_layer = len(self.W) - 1
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            pre_activations.append(z)
            a = softmax(z) if i == last_layer else relu(z)
            activations.append(a)
        return pre_activations, activations

    def predict_proba(self, X):
        """Returns the (n, 4) matrix of action probabilities."""
        _, activations = self.forward(X)
        return activations[-1]

    def predict(self, X):
        """Returns the (n,) array of predicted action labels (0-3)."""
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y) * 100.0

    def backward(self, pre_activations, activations, y):
        """Computes the gradient of the cross-entropy loss w.r.t. every
        weight and bias, using standard backpropagation."""
        n = len(y)
        num_layers = len(self.W)
        grad_W = [None] * num_layers
        grad_b = [None] * num_layers

        # For softmax + cross-entropy, the gradient at the output layer
        # is simply (predicted probabilities - one-hot true label).
        delta = activations[-1].copy()
        delta[np.arange(n), y] -= 1.0
        delta /= n

        for i in reversed(range(num_layers)):
            grad_W[i] = activations[i].T @ delta
            grad_b[i] = np.sum(delta, axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * relu_derivative(pre_activations[i - 1])

        return grad_W, grad_b

    def train(self, X, y, epochs=200, batch_size=32, lr=0.1, verbose=True):
        """Trains with simple mini-batch gradient descent."""
        rng = np.random.default_rng(SEED)
        n = len(X)

        for epoch in range(epochs):
            perm = rng.permutation(n)
            X_shuffled, y_shuffled = X[perm], y[perm]

            for start in range(0, n, batch_size):
                xb = X_shuffled[start:start + batch_size]
                yb = y_shuffled[start:start + batch_size]

                pre_activations, activations = self.forward(xb)
                grad_W, grad_b = self.backward(pre_activations, activations, yb)

                for i in range(len(self.W)):
                    self.W[i] -= lr * grad_W[i]
                    self.b[i] -= lr * grad_b[i]

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                acc = self.accuracy(X, y)
                print(f"  epoch {epoch:3d}  train_acc={acc:.2f}%")

    def save_weights_augmented(self, path):
        """Writes the three weight matrices in the format the assignment
        wants: bias folded in as the last row of each matrix, matrices
        separated by a line of 5 dashes (matches A1's q5_weights.txt
        convention), rows separated by newlines, columns separated by
        commas, values rounded to 4 decimals."""
        blocks = []
        for W, b in zip(self.W, self.b):
            augmented = np.vstack([W, b.reshape(1, -1)])  # shape (fan_in + 1, fan_out)
            lines = [",".join(f"{v:.4f}" for v in row) for row in augmented]
            blocks.append("\n".join(lines))
        with open(path, "w") as f:
            f.write("\n-----\n".join(blocks))
