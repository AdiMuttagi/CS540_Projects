"""
Question 4 (Part 1): for each of the 300 rows, pick the action with the
highest probability from Question 3's stochastic policy output.

Question 5 (Part 2): run the ACTUAL behavior policy rule (not the
network) on the same 300 feature rows from Question 1, to get the
"ground truth" action for each row.

Both outputs are 300 numbers, comma separated, on a single line.
"""
import os
import numpy as np
from behavior_policy import behavior_actions

root = os.path.dirname(os.path.abspath(__file__))

# --- Question 4: argmax of the stochastic policy from Question 3 ---
probs = np.loadtxt(os.path.join(root, "answers", "q3_probs.txt"), delimiter=",", dtype=np.float64)
best_actions = np.argmax(probs, axis=1)

q4_path = os.path.join(root, "answers", "q4_actions.txt")
with open(q4_path, "w") as f:
    f.write(",".join(str(a) for a in best_actions))
print(f"Wrote {len(best_actions)} actions to {q4_path}")

# --- Question 5: behavior policy applied to the Question 1 feature matrix ---
sensors = np.loadtxt(os.path.join(root, "answers", "q1_sensors.txt"), delimiter=",", dtype=np.float64)
true_actions = behavior_actions(sensors)

q5_path = os.path.join(root, "answers", "q5_behavior_actions.txt")
with open(q5_path, "w") as f:
    f.write(",".join(str(a) for a in true_actions))
print(f"Wrote {len(true_actions)} actions to {q5_path}")
