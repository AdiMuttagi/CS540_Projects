import os
import numpy as np

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
SEED = 42

# Maps the network's output (0, 1, 2) back to shape names.
CLASS_NAMES = {0: "sphere", 1: "cube", 2: "tetrahedral"}


# ---------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------
def relu(z):
    return np.maximum(0.0, z)


def softmax(z):
    # Subtract the row max for numerical stability, then normalize.
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def cross_entropy_loss(probs, y):
    n = len(y)
    return -np.mean(np.log(probs[np.arange(n), y] + 1e-8))


# ---------------------------------------------------------------------
# The network: fully connected, ReLU hidden layers, softmax output.
# Architecture is set by `sizes`, e.g. [1024, 128, 128, 3].
# Trained with mini-batch gradient descent + the Adam optimizer and a
# small L2 weight penalty. All written from scratch in NumPy.
# ---------------------------------------------------------------------
class MLP:
    def __init__(self, sizes, l2=1e-4, seed=SEED):
        rng = np.random.default_rng(seed)
        self.l2 = l2
        self.W = []
        self.b = []
        # One weight matrix + bias vector per layer.
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            # He initialization, the right scale for ReLU layers.
            self.W.append(rng.normal(0.0, np.sqrt(2.0 / fan_in), (fan_in, fan_out)))
            self.b.append(np.zeros(fan_out))
        # Adam keeps a running average of gradients (m) and squared
        # gradients (v) for every weight and bias.
        self.mW = [np.zeros_like(w) for w in self.W]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0  # Adam time step

    def forward(self, X):
        """Run inputs through the network, saving every intermediate
        value so backprop can reuse them."""
        pre_activations = []   # values before ReLU/softmax
        activations = [X]      # values after ReLU/softmax (X is layer 0)
        a = X
        last = len(self.W) - 1
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            pre_activations.append(z)
            a = softmax(z) if i == last else relu(z)
            activations.append(a)
        return pre_activations, activations

    def backward(self, pre_activations, activations, y):
        """Compute the gradient of the loss for every weight and bias."""
        n = len(y)
        L = len(self.W)
        grad_W = [None] * L
        grad_b = [None] * L

        # Gradient at the output layer for softmax + cross-entropy
        # simplifies to (predicted_probs - true_one_hot).
        delta = activations[-1].copy()
        delta[np.arange(n), y] -= 1.0
        delta /= n

        # Walk backwards through the layers.
        for i in reversed(range(L)):
            grad_W[i] = activations[i].T @ delta + self.l2 * self.W[i]
            grad_b[i] = np.sum(delta, axis=0)
            if i > 0:
                # Propagate the error back, then apply the ReLU
                # derivative (1 where the input was positive, else 0).
                delta = (delta @ self.W[i].T) * (pre_activations[i - 1] > 0)
        return grad_W, grad_b

    def adam_step(self, grad_W, grad_b, lr, b1=0.9, b2=0.999, eps=1e-8):
        """Update every weight and bias using the Adam rule."""
        self.t += 1
        correction1 = 1 - b1 ** self.t
        correction2 = 1 - b2 ** self.t
        for i in range(len(self.W)):
            self.mW[i] = b1 * self.mW[i] + (1 - b1) * grad_W[i]
            self.vW[i] = b2 * self.vW[i] + (1 - b2) * (grad_W[i] ** 2)
            self.W[i] -= lr * (self.mW[i] / correction1) / (np.sqrt(self.vW[i] / correction2) + eps)

            self.mb[i] = b1 * self.mb[i] + (1 - b1) * grad_b[i]
            self.vb[i] = b2 * self.vb[i] + (1 - b2) * (grad_b[i] ** 2)
            self.b[i] -= lr * (self.mb[i] / correction1) / (np.sqrt(self.vb[i] / correction2) + eps)

    def predict_proba(self, X):
        return self.forward(X)[1][-1]

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y) * 100.0

    def train(self, X, y, X_val, y_val,
              epochs=60, batch_size=256, lr=1e-3, patience=8):
        rng = np.random.default_rng(SEED)
        n = len(X)
        best_val = -1.0
        best_weights = None
        wait = 0

        for epoch in range(epochs):
            # Shuffle the data each epoch, then go through it in
            # mini-batches of `batch_size`.
            perm = rng.permutation(n)
            X_shuf, y_shuf = X[perm], y[perm]
            for start in range(0, n, batch_size):
                xb = X_shuf[start:start + batch_size]
                yb = y_shuf[start:start + batch_size]
                pre, acts = self.forward(xb)
                grad_W, grad_b = self.backward(pre, acts, yb)
                self.adam_step(grad_W, grad_b, lr)

            val_acc = self.accuracy(X_val, y_val)
            if epoch % 5 == 0 or epoch == epochs - 1:
                loss = cross_entropy_loss(self.predict_proba(X), y)
                print(f"  epoch {epoch:02d}  loss={loss:.4f}  val_acc={val_acc:.2f}%")

            # Early stopping: remember the best model on the validation
            # set and stop once it stops improving.
            if val_acc > best_val:
                best_val = val_acc
                best_weights = ([w.copy() for w in self.W], [b.copy() for b in self.b])
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"  early stop at epoch {epoch} (best val {best_val:.2f}%)")
                    break

        # Restore the best weights we saw.
        if best_weights is not None:
            self.W, self.b = best_weights


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def load_dataset(path):
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if data.ndim == 1:          # a single-row file loads as 1D; make it 2D
        data = data[np.newaxis, :]
    return data


def export_weights(model, filename="q6_weights.txt"):
    """Write the three weight matrices, biases as each matrix's last row,
    separated by a line of dashes, values comma-separated, 4 decimals."""
    blocks = []
    for W, b in zip(model.W, model.b):
        M = np.vstack([W, b])                       # bias becomes the last row
        rows = [",".join("0.0000" if f"{v:.4f}" == "-0.0000" else f"{v:.4f}"
                         for v in row) for row in M]
        blocks.append("\n".join(rows))
    with open(filename, "w") as f:
        f.write("\n-----\n".join(blocks))
    print(f"Wrote {filename}")


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "data")

    print("Loading data...")
    files = {0: "sphere.txt", 1: "cube.txt", 2: "tetrahedral.txt"}
    X_list, y_list = [], []
    for label, fname in files.items():
        arr = load_dataset(os.path.join(data_dir, fname))
        X_list.append(arr)
        y_list.append(np.full(len(arr), label, np.int64))
        print(f"  {fname}: {len(arr)} images")

    # Stack all shapes together and scale pixels to the 0-1 range.
    X = np.vstack(X_list).astype(np.float64) / 255.0
    y = np.concatenate(y_list)

    # Shuffle, then split 90% train / 10% validation.
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    n_train = int(0.9 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    # Standardize features using the TRAINING set's mean and std only,
    # then apply the same transform to the validation set (no peeking).
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    # Build and train: 1024 inputs -> 128 -> 128 -> 3 outputs.
    model = MLP([X_train.shape[1], 128, 128, 3], l2=1e-4)
    model.train(X_train, y_train, X_val, y_val,
                epochs=60, batch_size=256, lr=1e-3, patience=8)

    print(f"\nTrain accuracy: {model.accuracy(X_train, y_train):.2f}%")
    print(f"Val   accuracy: {model.accuracy(X_val, y_val):.2f}%")

    # Write the weight matrices for the assignment (Question 5).
    export_weights(model, os.path.join(root, "q6_weights.txt"))


if __name__ == "__main__":
    main()