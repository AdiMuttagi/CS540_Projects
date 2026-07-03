"""
Part 1: build a policy network with random (untrained) weights and make
sure it can take the vehicle's state and produce a valid action.

This does NOT check for correct behavior (the weights are random, so the
actions won't match the behavior policy) -- it just proves the network's
forward pass runs and produces one of the 4 actions for every input row.
"""
import os
import numpy as np
from policy_network import PolicyNetwork

ACTION_NAMES = {0: "turn left", 1: "turn right", 2: "speed up", 3: "no action"}


def load_data(path):
    """Each row is: action, x, y, vx, vy, s1, ..., sk.
    The network's input is only the sensors (s1..sk)."""
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    y = data[:, 0].astype(int)
    X = data[:, 5:]  # sensors only
    return X, y


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    X, y = load_data(os.path.join(root, "data", "test.txt"))

    k = X.shape[1]
    print(f"Loaded {len(X)} rows, k={k} sensors")

    net = PolicyNetwork(k=k, hidden1=32, hidden2=32)

    probs = net.predict_proba(X)
    actions = net.predict(X)

    print("\nFirst 5 rows: predicted action probabilities and chosen action")
    for i in range(5):
        probs_str = ", ".join(f"{p:.3f}" for p in probs[i])
        print(f"  row {i}: probs=[{probs_str}]  -> {ACTION_NAMES[actions[i]]}")

    out_path = os.path.join(root, "answers", "part1_actions.txt")
    np.savetxt(out_path, actions, fmt="%d")
    print(f"\nSaved all {len(actions)} predicted actions to {out_path}")


if __name__ == "__main__":
    main()
