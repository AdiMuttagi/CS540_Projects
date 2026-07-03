"""
Question 6 (Part 2): train the network on our training set to clone the
behavior policy, then export the trained weights in the same format as
Question 2 (bias folded into the last row, matrices separated by -----).
"""
import os
import numpy as np
from policy_network import PolicyNetwork

root = os.path.dirname(os.path.abspath(__file__))


def load_data(path):
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    y = data[:, 0].astype(int)
    X = data[:, 5:]  # sensors only
    return X, y


X, y = load_data(os.path.join(root, "data", "test.txt"))
k = X.shape[1]
print(f"Loaded {len(X)} rows, k={k} sensors")

net = PolicyNetwork(k=k, hidden1=32, hidden2=32)
print(f"train_acc before training: {net.accuracy(X, y):.2f}%")

net.train(X, y, epochs=600, batch_size=32, lr=0.05)
print(f"train_acc after training: {net.accuracy(X, y):.2f}%")

q6_path = os.path.join(root, "answers", "q6_weights.txt")
net.save_weights_augmented(q6_path)
print(f"Wrote trained weights to {q6_path}")
