import numpy as np
import warnings
import time

warnings.filterwarnings("ignore")

with open("data.txt", "r") as f:
    lines = f.read().strip().split("\n")

X = []
for line in lines:
    pixels = [int(x.strip()) for x in line.split(",")]
    X.append(pixels)

X = np.array(X)
X = X.astype(np.float64) / 255.0

y = []
for i in range(len(X)):
    y.append(i % 3)

y = np.array(y)

X_train = X[:240]
y_train = y[:240]

X_test = X[240:]
y_test = y[240:]

# Network architecture
input_size = 1024
hidden_size = 128
output_size = 3

np.random.seed(42)

W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros(hidden_size)

W2 = np.random.randn(hidden_size, hidden_size) * 0.01
b2 = np.zeros(hidden_size)

W3 = np.random.randn(hidden_size, output_size) * 0.01
b3 = np.zeros(output_size)

def relu(z):
    return np.maximum(0, z)

def forward_pass(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    return z1, a1, z2, a2, z3

def softmax(z):
    e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e_z / e_z.sum(axis=1, keepdims=True)

def cross_entropy_loss(output, y):
    n = len(y)
    correct_probs = output[np.arange(n), y]
    loss = -np.sum(np.log(correct_probs + 1e-8)) / n
    return loss

def backprop(X, y, z1, a1, z2, a2, z3, output):
    n = len(y)
    dz3 = output.copy()
    dz3[np.arange(n), y] -= 1
    dz3 /= n
    dW3 = a2.T @ dz3
    db3 = np.sum(dz3, axis=0)
    dz2 = (dz3 @ W3.T) * (z2 > 0)
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)
    dz1 = (dz2 @ W2.T) * (z1 > 0)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0)
    return dW1, db1, dW2, db2, dW3, db3

def gradient_descent(dW1, db1, dW2, db2, dW3, db3, learning_rate):
    global W1, b1, W2, b2, W3, b3
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3

learning_rate = 0.01
epochs = 2000

start = time.time()

for epoch in range(epochs):
    z1, a1, z2, a2, z3 = forward_pass(X_train)
    output = softmax(z3)
    loss = cross_entropy_loss(output, y_train)
    dW1, db1, dW2, db2, dW3, db3 = backprop(X_train, y_train, z1, a1, z2, a2, z3, output)
    gradient_descent(dW1, db1, dW2, db2, dW3, db3, learning_rate)

    if epoch % 100 == 0:
        elapsed = time.time() - start
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Time: {elapsed:.1f}s")

def predict(X):
    _, _, _, _, z3 = forward_pass(X)
    output = softmax(z3)
    return np.argmax(output, axis=1)

train_preds = predict(X_train)
train_accuracy = np.mean(train_preds == y_train) * 100
print(f"Training accuracy: {train_accuracy:.2f}%")

test_preds = predict(X_test)
test_accuracy = np.mean(test_preds == y_test) * 100
print(f"Test accuracy: {test_accuracy:.2f}%")