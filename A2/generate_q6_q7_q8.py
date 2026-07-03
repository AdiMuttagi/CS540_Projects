"""
Question 6 (Part 2): train the network on our training set to clone the
behavior policy, export the trained weights.

Question 7 (Part 2): run that SAME trained network on the Question 1
feature matrix, output the stochastic policy (300 lines x 4 probabilities).

Question 8 (Part 2): pick the highest-probability action from Question 7
for each row -- these should match the behavior policy actions (Question 5).

All three questions must use the same trained network, so we train once
and reuse it for every output.
"""
import os
import numpy as np
from policy_network import PolicyNetwork
from behavior_policy import behavior_actions

root = os.path.dirname(os.path.abspath(__file__))


def load_data(path):
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    y = data[:, 0].astype(int)
    X = data[:, 5:]  # sensors only
    return X, y


# --- Question 6: train and export weights ---
X, y = load_data(os.path.join(root, "data", "test.txt"))
k = X.shape[1]
print(f"Loaded {len(X)} rows, k={k} sensors")

net = PolicyNetwork(k=k, hidden1=64, hidden2=64)
print(f"train_acc before training: {net.accuracy(X, y):.2f}%")

net.train(X, y, epochs=1500, batch_size=32, lr=0.05)
print(f"train_acc after training: {net.accuracy(X, y):.2f}%")

q6_path = os.path.join(root, "answers", "q6_weights.txt")
net.save_weights_augmented(q6_path)
print(f"Wrote trained weights to {q6_path}")

# --- Question 7: stochastic policy on the Question 1 feature matrix ---
sensors = np.loadtxt(os.path.join(root, "answers", "q1_sensors.txt"), delimiter=",", dtype=np.float64)
probs = net.predict_proba(sensors)

q7_path = os.path.join(root, "answers", "q7_probs.txt")
with open(q7_path, "w") as f:
    for row in probs:
        f.write(",".join(f"{p:.4f}" for p in row) + "\n")
print(f"Wrote {len(probs)} rows of action probabilities to {q7_path}")

# --- Question 8: argmax of Question 7's probabilities ---
best_actions = np.argmax(probs, axis=1)

q8_path = os.path.join(root, "answers", "q8_actions.txt")
with open(q8_path, "w") as f:
    f.write(",".join(str(a) for a in best_actions))
print(f"Wrote {len(best_actions)} actions to {q8_path}")

# Sanity check: how well does this match the behavior policy (Question 5)?
true_actions = behavior_actions(sensors)
match = np.mean(best_actions == true_actions) * 100.0
print(f"Question 8 actions match the behavior policy: {match:.2f}%")
