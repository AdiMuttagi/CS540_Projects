"""
Question 2 (Part 1): export a network with random weights, in the
format the quiz wants -- three weight matrices, bias folded in as the
last row of each, matrices separated by a line of dashes.

Question 3 (Part 1): run that SAME network on the Question 1 feature
matrix (the sensor readings) and output the softmax probabilities for
all 300 rows, 4 numbers per line, comma separated, rounded to 4 decimals.

Both questions must use the same random weights, so we build the
network once and reuse it for both outputs.
"""
import os
import numpy as np
from policy_network import PolicyNetwork

root = os.path.dirname(os.path.abspath(__file__))

# Same k, hidden sizes as the rest of Part 1/2 (5 sensors, hidden layers <= 100).
net = PolicyNetwork(k=5, hidden1=32, hidden2=32)

# --- Question 2: export the random weights ---
q2_path = os.path.join(root, "answers", "q2_weights.txt")
net.save_weights_augmented(q2_path)
print(f"Wrote random weights to {q2_path}")

# --- Question 3: run the network on the Question 1 feature matrix ---
sensors = np.loadtxt(os.path.join(root, "answers", "q1_sensors.txt"), delimiter=",", dtype=np.float64)
probs = net.predict_proba(sensors)

q3_path = os.path.join(root, "answers", "q3_probs.txt")
with open(q3_path, "w") as f:
    for row in probs:
        f.write(",".join(f"{p:.4f}" for p in row) + "\n")
print(f"Wrote {len(probs)} rows of action probabilities to {q3_path}")
